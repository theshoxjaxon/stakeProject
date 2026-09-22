"""Auth router: register, login, refresh, logout, me.

Mounted at /v1/auth by src/api/__init__.py.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.api.deps import Principal, current_principal, get_current_plan, get_db
from src.core.security import (
    RefreshTokenInvalidError,
    RefreshTokenReuseError,
    create_access_token,
    hash_password,
    hash_token,
    issue_refresh_token,
    rotate_refresh_token,
    verify_password,
)
from src.models import Organization, RefreshToken, User

logger = logging.getLogger(__name__)
router = APIRouter()


class RegisterRequest(BaseModel):
    org_name: str = Field(..., min_length=1)
    email: EmailStr
    password: str = Field(..., min_length=8)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class LogoutRequest(BaseModel):
    refresh_token: str


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class MeResponse(BaseModel):
    user_id: str
    org_id: str
    email: str
    role: str
    plan: str


def _issue_token_pair(db: Session, user: User, org: Organization) -> TokenPair:
    access = create_access_token(
        sub=str(user.id), org=str(org.id), role=user.role, plan=org.plan
    )
    raw_refresh, _ = issue_refresh_token(db, user.id)
    return TokenPair(access_token=access, refresh_token=raw_refresh)


@router.post("/register", response_model=TokenPair, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest, db: Session = Depends(get_db)) -> TokenPair:
    """Create an organization and its first (owner) user, and log them in."""
    existing = db.execute(
        select(User).where(User.email == body.email)
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(status_code=409, detail={"error": "email_taken"})

    org = Organization(name=body.org_name)
    db.add(org)
    db.flush()  # org.id populated (needed for user.org_id FK below)

    user = User(
        org_id=org.id,
        email=body.email,
        password_hash=hash_password(body.password),
        role="owner",
    )
    db.add(user)
    db.flush()  # user.id populated (needed for the access token + refresh token FK)

    pair = _issue_token_pair(db, user, org)
    db.commit()
    return pair


@router.post("/login", response_model=TokenPair)
def login(body: LoginRequest, db: Session = Depends(get_db)) -> TokenPair:
    user = db.execute(select(User).where(User.email == body.email)).scalar_one_or_none()
    if (
        user is None
        or not user.is_active
        or not verify_password(body.password, user.password_hash)
    ):
        raise HTTPException(status_code=401, detail={"error": "invalid_credentials"})

    org = db.get(Organization, user.org_id)
    pair = _issue_token_pair(db, user, org)
    db.commit()
    return pair


@router.post("/refresh", response_model=TokenPair)
def refresh(body: RefreshRequest, db: Session = Depends(get_db)) -> TokenPair:
    try:
        raw_new, new_token = rotate_refresh_token(db, body.refresh_token)
    except RefreshTokenReuseError as exc:
        # rotate_refresh_token already revoked the chain on this session;
        # persist that before responding.
        db.commit()
        logger.warning(
            "Refresh token reuse detected for user_id=%s (possible theft).", exc.user_id
        )
        raise HTTPException(
            status_code=401, detail={"error": "refresh_token_reused"}
        ) from exc
    except RefreshTokenInvalidError as exc:
        raise HTTPException(
            status_code=401, detail={"error": "invalid_refresh_token"}
        ) from exc

    user = db.get(User, new_token.user_id)
    if user is None or not user.is_active:
        db.rollback()
        raise HTTPException(status_code=401, detail={"error": "invalid_refresh_token"})

    org = db.get(Organization, user.org_id)
    access = create_access_token(
        sub=str(user.id), org=str(org.id), role=user.role, plan=org.plan
    )
    db.commit()
    return TokenPair(access_token=access, refresh_token=raw_new)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(body: LogoutRequest, db: Session = Depends(get_db)) -> None:
    """Revoke one refresh token. Idempotent: always 204, even if unknown."""
    token_hash = hash_token(body.refresh_token)
    token = db.execute(
        select(RefreshToken).where(RefreshToken.token_hash == token_hash)
    ).scalar_one_or_none()
    if token is not None and token.revoked_at is None:
        token.revoked_at = datetime.now(timezone.utc)
        db.commit()
    return None


@router.get("/me", response_model=MeResponse)
def me(
    principal: Principal = Depends(current_principal),
    plan: str = Depends(get_current_plan),
    db: Session = Depends(get_db),
) -> MeResponse:
    """The authenticated dashboard user's profile — not available for API keys."""
    if principal.user_id is None:
        raise HTTPException(
            status_code=403, detail={"error": "not_available_for_api_keys"}
        )

    user = db.get(User, uuid.UUID(principal.user_id))
    if user is None:
        raise HTTPException(status_code=401, detail={"error": "invalid_token"})

    return MeResponse(
        user_id=str(user.id),
        org_id=str(user.org_id),
        email=user.email,
        role=user.role,
        plan=plan,
    )
