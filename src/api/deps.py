"""FastAPI dependencies: authentication, authorization, plan gating, rate limiting.

- current_principal: resolves EITHER a dashboard JWT OR a server-to-server
  ``sk_live_`` API key into one Principal.
- require_role(*roles): 403 if the principal's role isn't one of these.
- require_plan(minimum): 402 if the org's *current* plan (re-read from the
  DB, Redis-cached for 60s — the JWT's plan claim is never trusted) ranks
  below ``minimum``.
- enforce_daily_rate_limit: 429 once the org's plan-based daily quota is
  exhausted, with X-RateLimit-* and Retry-After headers.

401 is used for "who are you" (missing/invalid credential) — distinct from
402/403/429, which assume a resolved principal and gate what it can do.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import jwt
import redis as redis_lib
from fastapi import Depends, HTTPException, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.config import DATABASE_URL, REDIS_URL
from src.core.security import API_KEY_PREFIX, decode_access_token, hash_token
from src.database import get_engine
from src.models import ApiKey, Organization

_PLAN_RANK = {"free": 0, "pro": 1, "premium": 2}
_PLAN_CACHE_TTL_SECONDS = 60

_DAILY_RATE_LIMITS = {"free": 100, "pro": 10_000, "premium": 100_000}
# Key TTL outlives one UTC day so a key created just before midnight is still
# cleaned up automatically, without depending on a second write to set it.
_RATE_LIMIT_KEY_TTL_SECONDS = 2 * 24 * 60 * 60

# ---------------------------------------------------------------------------
# DB / Redis — shared, pooled clients. src/api/__init__.py imports these
# rather than creating its own, so there's exactly one engine and one Redis
# connection pool per process.
# ---------------------------------------------------------------------------

db_engine = get_engine(DATABASE_URL)


def get_db():
    """Request-scoped SQLAlchemy session (sync). Safe for lazy-loaded relationships."""
    with Session(db_engine) as session:
        yield session


_redis_client = redis_lib.from_url(REDIS_URL, decode_responses=True)


def get_redis() -> redis_lib.Redis:
    """Shared Redis client (connection-pooled by redis-py itself)."""
    return _redis_client


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Principal:
    """One resolved caller, however they authenticated."""

    user_id: str | None  # None for API-key callers — keys aren't tied to a user
    org_id: str
    role: str
    scopes: list[str]


_bearer_scheme = HTTPBearer(auto_error=False)


def current_principal(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    db: Session = Depends(get_db),
) -> Principal:
    """Resolve the bearer credential to a Principal — a JWT or an sk_live_ API key."""
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail={"error": "not_authenticated"})

    token = credentials.credentials
    if token.startswith(API_KEY_PREFIX):
        return _resolve_api_key_principal(db, token)
    return _resolve_jwt_principal(token)


def _resolve_jwt_principal(token: str) -> Principal:
    try:
        claims = decode_access_token(token)
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail={"error": "invalid_token"}) from exc

    # Dashboard users have no API scopes — scopes are an API-key concept.
    return Principal(
        user_id=claims["sub"], org_id=claims["org"], role=claims["role"], scopes=[]
    )


def _resolve_api_key_principal(db: Session, raw_key: str) -> Principal:
    key_hash = hash_token(raw_key)
    api_key = db.execute(
        select(ApiKey).where(ApiKey.key_hash == key_hash)
    ).scalar_one_or_none()

    if api_key is None or api_key.revoked_at is not None:
        raise HTTPException(status_code=401, detail={"error": "invalid_api_key"})

    api_key.last_used_at = datetime.now(timezone.utc)
    db.flush()

    # API keys have no user and no role hierarchy of their own — authorized
    # purely by scopes, so they get the least-privileged role by default.
    return Principal(
        user_id=None,
        org_id=str(api_key.org_id),
        role="member",
        scopes=list(api_key.scopes or []),
    )


# ---------------------------------------------------------------------------
# require_role
# ---------------------------------------------------------------------------


def require_role(*roles: str):
    """Dependency factory: 403 unless the principal's role is one of `roles`."""

    def _dependency(principal: Principal = Depends(current_principal)) -> Principal:
        if principal.role not in roles:
            raise HTTPException(
                status_code=403,
                detail={"error": "forbidden", "allowed_roles": list(roles)},
            )
        return principal

    return _dependency


# ---------------------------------------------------------------------------
# require_plan
# ---------------------------------------------------------------------------


def _effective_org_plan(db: Session, redis_client: redis_lib.Redis, org_id: str) -> str:
    """
    The org's plan right now — DB is the source of truth, Redis is a 60s cache.

    Never derived from a JWT claim: a plan claim goes stale the instant an
    org downgrades, and a still-valid access token could outlive that by up
    to its full 15-minute TTL.
    """
    cache_key = f"org_plan:{org_id}"
    cached = redis_client.get(cache_key)
    if cached is not None:
        return cached

    org = db.get(Organization, uuid.UUID(org_id))
    if org is None:
        plan = "free"
    else:
        expires_at = org.plan_expires_at
        if expires_at is not None and _as_utc(expires_at) <= datetime.now(timezone.utc):
            plan = "free"
        else:
            plan = org.plan

    redis_client.set(cache_key, plan, ex=_PLAN_CACHE_TTL_SECONDS)
    return plan


def _as_utc(dt: datetime) -> datetime:
    """Normalize to aware UTC (SQLite round-trips tz-aware columns as naive)."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def get_current_plan(
    principal: Principal = Depends(current_principal),
    db: Session = Depends(get_db),
    redis_client: redis_lib.Redis = Depends(get_redis),
) -> str:
    """
    The caller's current plan, for FIELD-level tier gating inside a response.

    Distinct from require_plan: this never raises. Routes that gate by field
    (not by endpoint) depend on this instead of require_plan.
    """
    return _effective_org_plan(db, redis_client, principal.org_id)


def require_plan(minimum: str):
    """Dependency factory: 402 unless the org's current plan ranks >= minimum."""
    minimum_rank = _PLAN_RANK[minimum]

    def _dependency(
        principal: Principal = Depends(current_principal),
        db: Session = Depends(get_db),
        redis_client: redis_lib.Redis = Depends(get_redis),
    ) -> Principal:
        plan = _effective_org_plan(db, redis_client, principal.org_id)
        # An unrecognized cached plan value (e.g. a stale/renamed tier) never
        # grants access — only a known, sufficiently-ranked plan does.
        if _PLAN_RANK.get(plan, 0) < minimum_rank:
            raise HTTPException(
                status_code=402,
                detail={"error": "upgrade_required", "required_plan": minimum},
            )
        return principal

    return _dependency


# ---------------------------------------------------------------------------
# enforce_daily_rate_limit
# ---------------------------------------------------------------------------


def _seconds_until_next_utc_midnight() -> int:
    now = datetime.now(timezone.utc)
    tomorrow = (now + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return int((tomorrow - now).total_seconds())


def enforce_daily_rate_limit(
    response: Response,
    principal: Principal = Depends(current_principal),
    db: Session = Depends(get_db),
    redis_client: redis_lib.Redis = Depends(get_redis),
) -> Principal:
    """429 once the org's plan-based daily request quota is exhausted."""
    plan = _effective_org_plan(db, redis_client, principal.org_id)
    limit = _DAILY_RATE_LIMITS.get(plan, _DAILY_RATE_LIMITS["free"])

    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    key = f"rl:{principal.org_id}:{day}"

    # INCR + conditional EXPIRE in one round trip: atomic, and idempotent
    # (NX means only the request that creates the key sets its TTL).
    pipe = redis_client.pipeline()
    pipe.incr(key)
    pipe.expire(key, _RATE_LIMIT_KEY_TTL_SECONDS, nx=True)
    count, _ = pipe.execute()

    if count > limit:
        raise HTTPException(
            status_code=429,
            detail={"error": "rate_limit_exceeded", "limit": limit},
            headers={
                "X-RateLimit-Limit": str(limit),
                "X-RateLimit-Remaining": "0",
                "Retry-After": str(_seconds_until_next_utc_midnight()),
            },
        )

    response.headers["X-RateLimit-Limit"] = str(limit)
    response.headers["X-RateLimit-Remaining"] = str(max(0, limit - count))
    return principal
