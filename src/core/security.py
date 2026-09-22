"""Authentication primitives: password hashing, access tokens, refresh tokens.

No HTTP endpoints here — this module only implements the building blocks:

- Passwords are hashed with argon2 (via passlib).
- Access tokens are short-lived JWTs (HS256, 15 min TTL).
- Refresh tokens are opaque secrets, never JWTs. Only their sha256 hash is
  ever persisted; the raw value is returned once, at issuance/rotation time,
  for the caller to hand to the client.
- Refresh tokens rotate on every use. Presenting a token that has already
  been rotated (i.e. already has ``replaced_by`` set) is treated as reuse —
  a signal the token was stolen — and revokes every live refresh token for
  that user.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone

import jwt
from dotenv import load_dotenv
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.models import RefreshToken

load_dotenv()

JWT_SECRET = os.environ.get("JWT_SECRET", "").strip()
if not JWT_SECRET:
    raise RuntimeError(
        "JWT_SECRET environment variable is not set. Refusing to start: "
        "access tokens cannot be signed without it."
    )

JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_TTL = timedelta(minutes=15)
REFRESH_TOKEN_TTL = timedelta(days=30)

_pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------


def hash_password(password: str) -> str:
    """Hash a plaintext password with argon2. Never store the plaintext."""
    return _pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Check a plaintext password against a previously hashed value."""
    return _pwd_context.verify(password, password_hash)


# ---------------------------------------------------------------------------
# Access tokens (JWT, HS256, 15 min)
# ---------------------------------------------------------------------------


def create_access_token(*, sub: str, org: str, role: str, plan: str) -> str:
    """Issue a signed access token for one user/org/role/plan combination."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": sub,
        "org": org,
        "role": role,
        "plan": plan,
        "typ": "access",
        "jti": str(uuid.uuid4()),
        "iat": now,
        "exp": now + ACCESS_TOKEN_TTL,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """
    Verify signature and expiry, return the claims.

    Raises (from PyJWT) on any invalid token: jwt.ExpiredSignatureError for
    an expired token, jwt.InvalidSignatureError for a tampered one, or
    jwt.InvalidTokenError more generally for a malformed one.
    """
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])


# ---------------------------------------------------------------------------
# Refresh tokens (opaque, rotated, reuse-detected)
# ---------------------------------------------------------------------------


class RefreshTokenError(Exception):
    """Base class for refresh-token failures."""


class RefreshTokenInvalidError(RefreshTokenError):
    """The presented value doesn't match any live, unrevoked refresh token."""


class RefreshTokenReuseError(RefreshTokenError):
    """
    A refresh token that was already rotated was presented again.

    Treated as theft: every live refresh token for the owning user has
    already been revoked by the time this is raised. Carries user_id so
    callers can log which account was affected without re-querying.
    """

    def __init__(self, user_id, message: str = "Refresh token reuse detected."):
        self.user_id = user_id
        super().__init__(message)


def hash_token(raw: str) -> str:
    """sha256 hex digest of an opaque secret — the only form ever persisted."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def generate_refresh_token() -> tuple[str, str]:
    """A fresh opaque refresh secret and its sha256 hash. Store only the hash."""
    raw = secrets.token_urlsafe(48)
    return raw, hash_token(raw)


# Server-to-server B2B API keys — same opaque-secret/hash-only shape as
# refresh tokens, distinguished by a recognizable prefix (à la Stripe).
API_KEY_PREFIX = "sk_live_"


def generate_api_key() -> tuple[str, str]:
    """A fresh sk_live_ API key and its sha256 hash. Store only the hash."""
    raw = API_KEY_PREFIX + secrets.token_urlsafe(32)
    return raw, hash_token(raw)


def _as_utc(dt: datetime) -> datetime:
    """Normalize a datetime to aware UTC (SQLite round-trips tz-aware columns as naive)."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def issue_refresh_token(session: Session, user_id) -> tuple[str, RefreshToken]:
    """Create and persist a brand-new refresh token for user_id (start of a new chain)."""
    raw, hashed = generate_refresh_token()
    token = RefreshToken(
        id=uuid.uuid4(),
        user_id=user_id,
        token_hash=hashed,
        expires_at=datetime.now(timezone.utc) + REFRESH_TOKEN_TTL,
    )
    session.add(token)
    session.flush()
    return raw, token


def _revoke_chain_for_user(session: Session, user_id) -> None:
    """Revoke every live refresh token belonging to user_id — the theft response."""
    now = datetime.now(timezone.utc)
    stmt = select(RefreshToken).where(
        RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None)
    )
    for token in session.execute(stmt).scalars().all():
        token.revoked_at = now


def rotate_refresh_token(session: Session, raw_token: str) -> tuple[str, RefreshToken]:
    """
    Exchange a live refresh token for a new one, marking the old one replaced.

    Presenting a token that has already been rotated (``replaced_by`` already
    set) is treated as theft: every live refresh token for that user is
    revoked, then RefreshTokenReuseError is raised.
    """
    presented_hash = hash_token(raw_token)
    existing = session.execute(
        select(RefreshToken).where(RefreshToken.token_hash == presented_hash)
    ).scalar_one_or_none()

    if existing is None:
        raise RefreshTokenInvalidError("Unknown refresh token.")

    if existing.replaced_by is not None:
        _revoke_chain_for_user(session, existing.user_id)
        raise RefreshTokenReuseError(
            existing.user_id,
            "Refresh token reuse detected — the entire chain has been revoked.",
        )

    if existing.revoked_at is not None:
        raise RefreshTokenInvalidError("Refresh token has been revoked.")

    if _as_utc(existing.expires_at) <= datetime.now(timezone.utc):
        raise RefreshTokenInvalidError("Refresh token has expired.")

    raw_new, new_token = issue_refresh_token(session, existing.user_id)
    existing.replaced_by = new_token.id
    session.flush()
    return raw_new, new_token
