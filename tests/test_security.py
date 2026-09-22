"""Tests for authentication primitives (src/core/security.py)."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from sqlalchemy import create_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.dialects.postgresql import CITEXT
from sqlalchemy.orm import Session

from src.core.security import (
    API_KEY_PREFIX,
    JWT_SECRET,
    RefreshTokenReuseError,
    create_access_token,
    decode_access_token,
    generate_api_key,
    generate_refresh_token,
    hash_password,
    issue_refresh_token,
    rotate_refresh_token,
    verify_password,
)
from src.models import Base, Organization, RefreshToken, User


# In-memory SQLite can't render the Postgres-only CITEXT type used by
# User.email; this shim is test-only and never touches production code.
@compiles(CITEXT, "sqlite")
def _compile_citext_sqlite(element, compiler, **kw):
    return "VARCHAR"


@pytest.fixture()
def db_session() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(
        engine, tables=[Organization.__table__, User.__table__, RefreshToken.__table__]
    )
    with Session(engine) as session:
        yield session


@pytest.fixture()
def user(db_session: Session) -> User:
    org = Organization(id=uuid.uuid4(), name="Acme Corp")
    db_session.add(org)
    db_session.flush()
    u = User(
        id=uuid.uuid4(),
        org_id=org.id,
        email="owner@acme.test",
        password_hash="unused-in-these-tests",
        role="owner",
    )
    db_session.add(u)
    db_session.commit()
    return u


# --- Password hashing -------------------------------------------------------


def test_hash_password_roundtrip() -> None:
    hashed = hash_password("correct-horse-battery-staple")

    assert hashed.startswith("$argon2")
    assert verify_password("correct-horse-battery-staple", hashed) is True
    assert verify_password("wrong-password", hashed) is False


# --- Access tokens ------------------------------------------------------------


def test_create_and_decode_access_token_roundtrip() -> None:
    token = create_access_token(sub="user-1", org="org-1", role="owner", plan="pro")

    claims = decode_access_token(token)

    assert claims["sub"] == "user-1"
    assert claims["org"] == "org-1"
    assert claims["role"] == "owner"
    assert claims["plan"] == "pro"
    assert claims["typ"] == "access"
    assert "jti" in claims and claims["jti"]
    assert claims["exp"] - claims["iat"] == 15 * 60


def test_expired_access_token_rejected() -> None:
    now = datetime.now(timezone.utc)
    expired_payload = {
        "sub": "user-1",
        "org": "org-1",
        "role": "owner",
        "plan": "pro",
        "typ": "access",
        "jti": str(uuid.uuid4()),
        "iat": int((now - timedelta(minutes=30)).timestamp()),
        "exp": int((now - timedelta(minutes=15)).timestamp()),
    }
    expired_token = jwt.encode(expired_payload, JWT_SECRET, algorithm="HS256")

    with pytest.raises(jwt.ExpiredSignatureError):
        decode_access_token(expired_token)


def test_tampered_access_token_signature_rejected() -> None:
    token = create_access_token(sub="user-1", org="org-1", role="owner", plan="pro")
    header, payload, signature = token.split(".")
    tampered_char = "a" if signature[0] != "a" else "b"
    tampered_signature = tampered_char + signature[1:]
    tampered_token = f"{header}.{payload}.{tampered_signature}"

    with pytest.raises(jwt.InvalidSignatureError):
        decode_access_token(tampered_token)


# --- Refresh tokens -----------------------------------------------------------


def test_generate_refresh_token_returns_raw_and_sha256_hash() -> None:
    raw, hashed = generate_refresh_token()

    assert raw != hashed
    assert hashed == hashlib.sha256(raw.encode("utf-8")).hexdigest()
    # secrets.token_urlsafe(48) is a distinct, non-JWT opaque secret.
    assert "." not in raw


def test_generate_api_key_has_prefix_and_matches_hash() -> None:
    raw, hashed = generate_api_key()

    assert raw.startswith(API_KEY_PREFIX)
    assert hashed == hashlib.sha256(raw.encode("utf-8")).hexdigest()


def test_refresh_token_reuse_revokes_entire_chain(
    db_session: Session, user: User
) -> None:
    raw_1, token_1 = issue_refresh_token(db_session, user.id)
    db_session.commit()

    # Legitimate rotation: 1 -> 2 -> 3.
    raw_2, token_2 = rotate_refresh_token(db_session, raw_1)
    db_session.commit()
    raw_3, token_3 = rotate_refresh_token(db_session, raw_2)
    db_session.commit()

    assert token_1.replaced_by == token_2.id
    assert token_2.replaced_by == token_3.id
    assert token_3.revoked_at is None

    # Attacker replays the already-rotated first token.
    with pytest.raises(RefreshTokenReuseError):
        rotate_refresh_token(db_session, raw_1)
    db_session.commit()

    for tok in (token_1, token_2, token_3):
        db_session.refresh(tok)
        assert tok.revoked_at is not None
