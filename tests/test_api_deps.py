"""Tests for FastAPI auth/authorization dependencies (src/api/deps.py)."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import fakeredis
import pytest
from fastapi import HTTPException, Response
from fastapi.security import HTTPAuthorizationCredentials
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session

from src.api.deps import (
    API_KEY_PREFIX,
    Principal,
    current_principal,
    enforce_daily_rate_limit,
    require_plan,
    require_role,
)
from src.core.security import create_access_token, hash_token
from src.models import ApiKey, Base, Organization


# In-memory SQLite can't render the Postgres-only JSONB type used by
# ApiKey.scopes; this shim is test-only and never touches production code.
@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(element, compiler, **kw):
    return "JSON"


@pytest.fixture()
def db_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[Organization.__table__, ApiKey.__table__])
    with Session(engine) as session:
        yield session


@pytest.fixture()
def redis_client() -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis(decode_responses=True)


def _make_org(
    db_session: Session, plan: str = "free", plan_expires_at=None
) -> Organization:
    org = Organization(
        id=uuid.uuid4(), name="Acme Corp", plan=plan, plan_expires_at=plan_expires_at
    )
    db_session.add(org)
    db_session.flush()
    return org


def _bearer(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


# --- current_principal: JWT and API key resolution --------------------------


def test_current_principal_resolves_valid_jwt(db_session: Session) -> None:
    token = create_access_token(sub="user-1", org="org-1", role="owner", plan="pro")

    principal = current_principal(credentials=_bearer(token), db=db_session)

    assert principal == Principal(
        user_id="user-1", org_id="org-1", role="owner", scopes=[]
    )


def test_current_principal_resolves_valid_api_key(db_session: Session) -> None:
    org = _make_org(db_session, plan="pro")
    raw_key = API_KEY_PREFIX + "testkeyvalue123"
    api_key = ApiKey(
        id=uuid.uuid4(),
        org_id=org.id,
        key_hash=hash_token(raw_key),
        prefix=raw_key[:12],
        scopes=["predictions:read"],
    )
    db_session.add(api_key)
    db_session.commit()

    principal = current_principal(credentials=_bearer(raw_key), db=db_session)

    assert principal.user_id is None
    assert principal.org_id == str(org.id)
    assert principal.role == "member"
    assert principal.scopes == ["predictions:read"]
    assert api_key.last_used_at is not None


def test_current_principal_rejects_missing_credential(db_session: Session) -> None:
    with pytest.raises(HTTPException) as exc_info:
        current_principal(credentials=None, db=db_session)

    assert exc_info.value.status_code == 401


def test_current_principal_rejects_invalid_jwt(db_session: Session) -> None:
    with pytest.raises(HTTPException) as exc_info:
        current_principal(credentials=_bearer("not-a-real-jwt"), db=db_session)

    assert exc_info.value.status_code == 401


def test_current_principal_rejects_unknown_api_key(db_session: Session) -> None:
    raw_key = API_KEY_PREFIX + "never-issued"

    with pytest.raises(HTTPException) as exc_info:
        current_principal(credentials=_bearer(raw_key), db=db_session)

    assert exc_info.value.status_code == 401


# --- require_role -------------------------------------------------------------


def test_require_role_allows_listed_role() -> None:
    principal = Principal(user_id="u1", org_id="o1", role="admin", scopes=[])

    result = require_role("owner", "admin")(principal=principal)

    assert result is principal


def test_require_role_rejects_unlisted_role() -> None:
    principal = Principal(user_id="u1", org_id="o1", role="member", scopes=[])

    with pytest.raises(HTTPException) as exc_info:
        require_role("owner", "admin")(principal=principal)

    assert exc_info.value.status_code == 403


# --- require_plan ---------------------------------------------------------------


def test_require_plan_allows_when_org_meets_minimum(
    db_session: Session, redis_client: fakeredis.FakeRedis
) -> None:
    org = _make_org(db_session, plan="premium")
    principal = Principal(user_id="u1", org_id=str(org.id), role="owner", scopes=[])

    result = require_plan("pro")(
        principal=principal, db=db_session, redis_client=redis_client
    )

    assert result is principal


def test_require_plan_blocks_when_org_below_minimum(
    db_session: Session, redis_client: fakeredis.FakeRedis
) -> None:
    org = _make_org(db_session, plan="free")
    principal = Principal(user_id="u1", org_id=str(org.id), role="owner", scopes=[])

    with pytest.raises(HTTPException) as exc_info:
        require_plan("pro")(
            principal=principal, db=db_session, redis_client=redis_client
        )

    assert exc_info.value.status_code == 402
    assert exc_info.value.detail == {
        "error": "upgrade_required",
        "required_plan": "pro",
    }


def test_require_plan_ignores_jwt_plan_claim_and_reads_db(
    db_session: Session, redis_client: fakeredis.FakeRedis
) -> None:
    """A JWT can claim plan=premium; require_plan must not trust it — only the DB counts."""
    org = _make_org(db_session, plan="free")
    token = create_access_token(
        sub="user-1", org=str(org.id), role="owner", plan="premium"
    )
    principal = current_principal(credentials=_bearer(token), db=db_session)

    with pytest.raises(HTTPException) as exc_info:
        require_plan("pro")(
            principal=principal, db=db_session, redis_client=redis_client
        )

    assert exc_info.value.status_code == 402


def test_require_plan_treats_expired_plan_as_free(
    db_session: Session, redis_client: fakeredis.FakeRedis
) -> None:
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    org = _make_org(db_session, plan="premium", plan_expires_at=yesterday)
    principal = Principal(user_id="u1", org_id=str(org.id), role="owner", scopes=[])

    with pytest.raises(HTTPException) as exc_info:
        require_plan("pro")(
            principal=principal, db=db_session, redis_client=redis_client
        )

    assert exc_info.value.status_code == 402


def test_require_plan_serves_cached_plan_until_cache_cleared(
    db_session: Session, redis_client: fakeredis.FakeRedis
) -> None:
    org = _make_org(db_session, plan="free")
    principal = Principal(user_id="u1", org_id=str(org.id), role="owner", scopes=[])
    redis_client.set(f"org_plan:{org.id}", "premium", ex=60)

    # DB says free, but the 60s cache says premium — cache wins.
    result = require_plan("pro")(
        principal=principal, db=db_session, redis_client=redis_client
    )
    assert result is principal

    redis_client.flushall()

    # Cache cleared: the real (free) plan from the DB is used, and this is denied.
    with pytest.raises(HTTPException) as exc_info:
        require_plan("pro")(
            principal=principal, db=db_session, redis_client=redis_client
        )
    assert exc_info.value.status_code == 402


# --- enforce_daily_rate_limit -----------------------------------------------------


def test_rate_limit_allows_under_limit_and_sets_headers(
    db_session: Session, redis_client: fakeredis.FakeRedis
) -> None:
    org = _make_org(db_session, plan="free")
    principal = Principal(user_id="u1", org_id=str(org.id), role="owner", scopes=[])
    response = Response()

    result = enforce_daily_rate_limit(
        response=response, principal=principal, db=db_session, redis_client=redis_client
    )

    assert result is principal
    assert response.headers["X-RateLimit-Limit"] == "100"
    assert response.headers["X-RateLimit-Remaining"] == "99"


def test_rate_limit_blocks_over_limit_with_429_and_headers(
    db_session: Session, redis_client: fakeredis.FakeRedis
) -> None:
    org = _make_org(db_session, plan="free")
    principal = Principal(user_id="u1", org_id=str(org.id), role="owner", scopes=[])

    # free plan's daily limit is 100 — exhaust it.
    for _ in range(100):
        enforce_daily_rate_limit(
            response=Response(),
            principal=principal,
            db=db_session,
            redis_client=redis_client,
        )

    with pytest.raises(HTTPException) as exc_info:
        enforce_daily_rate_limit(
            response=Response(),
            principal=principal,
            db=db_session,
            redis_client=redis_client,
        )

    exc = exc_info.value
    assert exc.status_code == 429
    assert exc.headers["X-RateLimit-Limit"] == "100"
    assert exc.headers["X-RateLimit-Remaining"] == "0"
    assert int(exc.headers["Retry-After"]) > 0
