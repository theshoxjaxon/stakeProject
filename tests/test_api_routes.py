"""HTTP-level tests for the /v1 API: auth, predictions, org routers, healthz/readyz.

One test per endpoint for status + response shape, through a real
TestClient (dependency_overrides swap the DB/Redis for SQLite + fakeredis).
Business-logic edge cases (tier fields, rotation reuse) already have
dedicated unit tests in test_serializers.py / test_security.py — these
tests exist to prove the routers are wired correctly, not to re-litigate
that logic.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import fakeredis
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import CITEXT, JSONB
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from src.api import app
from src.api.deps import get_db, get_redis
from src.core.security import generate_api_key, hash_password
from src.models import ApiKey, Base, Match, Organization, Prediction, RefreshToken, User


# In-memory SQLite can't render the Postgres-only CITEXT/JSONB types used by
# User.email / ApiKey.scopes; these shims are test-only and never touch
# production code.
@compiles(CITEXT, "sqlite")
def _compile_citext_sqlite(element, compiler, **kw):
    return "VARCHAR"


@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(element, compiler, **kw):
    return "JSON"


_TABLES = [
    Organization.__table__,
    User.__table__,
    ApiKey.__table__,
    RefreshToken.__table__,
    Match.__table__,
    Prediction.__table__,
]


@pytest.fixture()
def client():
    # StaticPool: all sessions must share the ONE underlying connection, or
    # each new connection gets its own separate, empty :memory: database.
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine, tables=_TABLES)
    TestSessionLocal = sessionmaker(bind=engine)

    def _override_get_db():
        session = TestSessionLocal()
        try:
            yield session
        finally:
            session.close()

    fake_redis = fakeredis.FakeRedis(decode_responses=True)

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_redis] = lambda: fake_redis
    try:
        # Deliberately not `with TestClient(app) as c:` — that would run the
        # real lifespan, which connects to the configured (real) Postgres to
        # fit the GoalEngine. None of the /v1 routes or healthz/readyz touch
        # app.state.goal_engine, so skipping lifespan is safe here.
        c = TestClient(app)
        c.test_sessionmaker = TestSessionLocal  # stash for setup helpers
        yield c
    finally:
        app.dependency_overrides.clear()


def _db(client) -> Session:
    return client.test_sessionmaker()


# --- auth: register / login / refresh / logout / me -------------------------


def test_register_creates_org_and_owner_and_returns_tokens(client) -> None:
    resp = client.post(
        "/v1/auth/register",
        json={
            "org_name": "Acme Corp",
            "email": "owner@acme.dev",
            "password": "correct-horse-1",
        },
    )

    assert resp.status_code == 201
    body = resp.json()
    assert "access_token" in body and "refresh_token" in body

    with _db(client) as session:
        user = session.query(User).filter_by(email="owner@acme.dev").one()
        assert user.role == "owner"
        org = session.get(Organization, user.org_id)
        assert org.name == "Acme Corp"
        assert org.plan == "free"


def test_register_rejects_duplicate_email(client) -> None:
    payload = {
        "org_name": "Acme",
        "email": "dupe@acme.dev",
        "password": "correct-horse-1",
    }
    client.post("/v1/auth/register", json=payload)

    resp = client.post("/v1/auth/register", json=payload)

    assert resp.status_code == 409


def test_login_with_correct_password_returns_tokens(client) -> None:
    client.post(
        "/v1/auth/register",
        json={
            "org_name": "Acme",
            "email": "user@acme.dev",
            "password": "correct-horse-1",
        },
    )

    resp = client.post(
        "/v1/auth/login", json={"email": "user@acme.dev", "password": "correct-horse-1"}
    )

    assert resp.status_code == 200
    assert "access_token" in resp.json()


def test_login_with_wrong_password_rejected(client) -> None:
    client.post(
        "/v1/auth/register",
        json={
            "org_name": "Acme",
            "email": "user2@acme.dev",
            "password": "correct-horse-1",
        },
    )

    resp = client.post(
        "/v1/auth/login", json={"email": "user2@acme.dev", "password": "wrong-password"}
    )

    assert resp.status_code == 401


def test_refresh_rotates_and_returns_new_pair(client) -> None:
    register = client.post(
        "/v1/auth/register",
        json={"org_name": "Acme", "email": "r@acme.dev", "password": "correct-horse-1"},
    )
    old_refresh = register.json()["refresh_token"]

    resp = client.post("/v1/auth/refresh", json={"refresh_token": old_refresh})

    assert resp.status_code == 200
    assert resp.json()["refresh_token"] != old_refresh


def test_refresh_reuse_of_rotated_token_rejected(client) -> None:
    register = client.post(
        "/v1/auth/register",
        json={
            "org_name": "Acme",
            "email": "reuse@acme.dev",
            "password": "correct-horse-1",
        },
    )
    old_refresh = register.json()["refresh_token"]
    client.post("/v1/auth/refresh", json={"refresh_token": old_refresh})

    resp = client.post("/v1/auth/refresh", json={"refresh_token": old_refresh})

    assert resp.status_code == 401


def test_logout_is_idempotent_204(client) -> None:
    register = client.post(
        "/v1/auth/register",
        json={
            "org_name": "Acme",
            "email": "lo@acme.dev",
            "password": "correct-horse-1",
        },
    )
    raw_refresh = register.json()["refresh_token"]

    first = client.post("/v1/auth/logout", json={"refresh_token": raw_refresh})
    second = client.post("/v1/auth/logout", json={"refresh_token": raw_refresh})

    assert first.status_code == 204
    assert second.status_code == 204


def test_me_returns_profile_for_jwt_principal(client) -> None:
    register = client.post(
        "/v1/auth/register",
        json={
            "org_name": "Acme",
            "email": "me@acme.dev",
            "password": "correct-horse-1",
        },
    )
    access = register.json()["access_token"]

    resp = client.get("/v1/auth/me", headers={"Authorization": f"Bearer {access}"})

    assert resp.status_code == 200
    assert resp.json()["email"] == "me@acme.dev"
    assert resp.json()["role"] == "owner"
    assert resp.json()["plan"] == "free"


def test_me_rejects_api_key_principal(client) -> None:
    client.post(
        "/v1/auth/register",
        json={
            "org_name": "Acme",
            "email": "apikeyorg@acme.dev",
            "password": "correct-horse-1",
        },
    )
    with _db(client) as session:
        org = session.query(User).filter_by(email="apikeyorg@acme.dev").one().org_id
        raw, hashed = generate_api_key()
        session.add(ApiKey(org_id=org, key_hash=hashed, prefix=raw[:12], scopes=[]))
        session.commit()

    resp = client.get("/v1/auth/me", headers={"Authorization": f"Bearer {raw}"})

    assert resp.status_code == 403


# --- predictions --------------------------------------------------------------


def _seed_prediction(
    client, *, kickoff_offset: timedelta, org_id=None
) -> tuple[str, int]:
    """Create a Match + Prediction and an access token for a fresh org/user."""
    email = f"{uuid.uuid4()}@acme.dev"
    register = client.post(
        "/v1/auth/register",
        json={"org_name": "Acme", "email": email, "password": "correct-horse-1"},
    )
    access = register.json()["access_token"]

    with _db(client) as session:
        match_id = str(uuid.uuid4())
        kickoff = datetime.now(timezone.utc) + kickoff_offset
        session.add(
            Match(
                id=match_id,
                date=kickoff,
                home_team="Arsenal",
                away_team="Chelsea",
                status="scheduled",
                sport_key="soccer_epl",
            )
        )
        prediction = Prediction(
            match_id=match_id,
            kickoff=kickoff.replace(tzinfo=None),
            league="soccer_epl",
            home_prob=0.5,
            draw_prob=0.3,
            away_prob=0.2,
        )
        session.add(prediction)
        session.commit()
        prediction_id = prediction.id

    return access, prediction_id


def test_predictions_today_lists_fixtures_in_next_24h(client) -> None:
    access, _ = _seed_prediction(client, kickoff_offset=timedelta(hours=2))

    resp = client.get(
        "/v1/predictions/today", headers={"Authorization": f"Bearer {access}"}
    )

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    assert body[0]["fixture"] == {"home_team": "Arsenal", "away_team": "Chelsea"}
    # free plan, future kickoff -> delayed, no probabilities yet.
    assert body[0]["probabilities"] is None


def test_predictions_today_excludes_fixtures_outside_window(client) -> None:
    access, _ = _seed_prediction(client, kickoff_offset=timedelta(days=3))

    resp = client.get(
        "/v1/predictions/today", headers={"Authorization": f"Bearer {access}"}
    )

    assert resp.status_code == 200
    assert resp.json() == []


def test_predictions_by_id_returns_serialized_prediction(client) -> None:
    access, prediction_id = _seed_prediction(client, kickoff_offset=timedelta(hours=1))

    resp = client.get(
        f"/v1/predictions/{prediction_id}",
        headers={"Authorization": f"Bearer {access}"},
    )

    assert resp.status_code == 200
    assert resp.json()["id"] == prediction_id


def test_predictions_by_id_404_for_unknown_id(client) -> None:
    access, _ = _seed_prediction(client, kickoff_offset=timedelta(hours=1))

    resp = client.get(
        "/v1/predictions/999999", headers={"Authorization": f"Bearer {access}"}
    )

    assert resp.status_code == 404


def test_predictions_requires_authentication(client) -> None:
    resp = client.get("/v1/predictions/today")

    assert resp.status_code == 401


# --- org: members, invite, api keys ---------------------------------------------


def test_list_members_includes_self(client) -> None:
    register = client.post(
        "/v1/auth/register",
        json={
            "org_name": "Acme",
            "email": "member1@acme.dev",
            "password": "correct-horse-1",
        },
    )
    access = register.json()["access_token"]

    resp = client.get("/v1/org/members", headers={"Authorization": f"Bearer {access}"})

    assert resp.status_code == 200
    emails = [m["email"] for m in resp.json()]
    assert "member1@acme.dev" in emails


def test_invite_member_by_owner_creates_pending_inactive_user(client) -> None:
    register = client.post(
        "/v1/auth/register",
        json={
            "org_name": "Acme",
            "email": "owner2@acme.dev",
            "password": "correct-horse-1",
        },
    )
    access = register.json()["access_token"]

    resp = client.post(
        "/v1/org/members/invite",
        json={"email": "invitee@acme.dev", "role": "member"},
        headers={"Authorization": f"Bearer {access}"},
    )

    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "pending"
    assert "password" not in body

    with _db(client) as session:
        invitee = session.query(User).filter_by(email="invitee@acme.dev").one()
        assert invitee.is_active is False


def test_invite_member_forbidden_for_plain_member(client) -> None:
    register = client.post(
        "/v1/auth/register",
        json={
            "org_name": "Acme",
            "email": "owner3@acme.dev",
            "password": "correct-horse-1",
        },
    )
    access_owner = register.json()["access_token"]
    with _db(client) as session:
        org_id = session.query(User).filter_by(email="owner3@acme.dev").one().org_id
        member = User(
            org_id=org_id,
            email="plainmember@acme.dev",
            password_hash=hash_password("correct-horse-1"),
            role="member",
        )
        session.add(member)
        session.commit()

    login = client.post(
        "/v1/auth/login",
        json={"email": "plainmember@acme.dev", "password": "correct-horse-1"},
    )
    access_member = login.json()["access_token"]

    resp = client.post(
        "/v1/org/members/invite",
        json={"email": "another@acme.dev", "role": "member"},
        headers={"Authorization": f"Bearer {access_member}"},
    )

    assert resp.status_code == 403
    assert access_owner  # sanity: owner token was obtained without error


def test_create_and_list_and_revoke_api_key(client) -> None:
    register = client.post(
        "/v1/auth/register",
        json={
            "org_name": "Acme",
            "email": "owner4@acme.dev",
            "password": "correct-horse-1",
        },
    )
    access = register.json()["access_token"]
    headers = {"Authorization": f"Bearer {access}"}

    create = client.post(
        "/v1/org/api-keys", json={"scopes": ["predictions:read"]}, headers=headers
    )
    assert create.status_code == 201
    created = create.json()
    assert created["api_key"].startswith("sk_live_")
    key_id = created["id"]

    listing = client.get("/v1/org/api-keys", headers=headers)
    assert listing.status_code == 200
    assert all("api_key" not in k and "key_hash" not in k for k in listing.json())
    assert any(k["id"] == key_id for k in listing.json())

    revoke = client.post(f"/v1/org/api-keys/{key_id}/revoke", headers=headers)
    assert revoke.status_code == 204

    listing_after = client.get("/v1/org/api-keys", headers=headers)
    revoked_entry = next(k for k in listing_after.json() if k["id"] == key_id)
    assert revoked_entry["revoked_at"] is not None


def test_api_key_can_authenticate_predictions_request(client) -> None:
    register = client.post(
        "/v1/auth/register",
        json={
            "org_name": "Acme",
            "email": "owner5@acme.dev",
            "password": "correct-horse-1",
        },
    )
    access = register.json()["access_token"]
    create = client.post(
        "/v1/org/api-keys",
        json={"scopes": ["predictions:read"]},
        headers={"Authorization": f"Bearer {access}"},
    )
    raw_key = create.json()["api_key"]

    resp = client.get(
        "/v1/predictions/today", headers={"Authorization": f"Bearer {raw_key}"}
    )

    assert resp.status_code == 200


# --- health --------------------------------------------------------------------


def test_healthz_is_always_ok(client) -> None:
    resp = client.get("/healthz")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_readyz_reports_db_and_redis_ok(client) -> None:
    resp = client.get("/readyz")

    assert resp.status_code == 200
    body = resp.json()
    assert body["db"] == "ok"
    assert body["redis"] == "ok"
