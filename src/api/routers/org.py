"""Org router: members, invites, API keys — everything scoped to the caller's org.

Mounted at /v1/org by src/api/__init__.py.

Reads (list members, list keys) are open to any authenticated member of the
org. Writes (invite, create key, revoke key) require owner/admin — an
authorization concern (require_role), unrelated to plan/tier gating.
"""

from __future__ import annotations

import secrets
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.api.deps import Principal, current_principal, get_db, require_role
from src.core.security import generate_api_key, hash_password
from src.models import ApiKey, User

router = APIRouter()


def _org_uuid(principal: Principal) -> uuid.UUID:
    return uuid.UUID(principal.org_id)


# --- members ------------------------------------------------------------------


class MemberOut(BaseModel):
    id: str
    email: str
    role: str
    is_active: bool


class InviteRequest(BaseModel):
    email: EmailStr
    role: str = Field(default="member")


class InviteResponse(BaseModel):
    user_id: str
    email: str
    role: str
    status: str = "pending"


@router.get("/members", response_model=list[MemberOut])
def list_members(
    principal: Principal = Depends(current_principal),
    db: Session = Depends(get_db),
) -> list[MemberOut]:
    users = (
        db.execute(select(User).where(User.org_id == _org_uuid(principal)))
        .scalars()
        .all()
    )
    return [
        MemberOut(id=str(u.id), email=u.email, role=u.role, is_active=u.is_active)
        for u in users
    ]


@router.post(
    "/members/invite",
    response_model=InviteResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_role("owner", "admin"))],
)
def invite_member(
    body: InviteRequest,
    principal: Principal = Depends(current_principal),
    db: Session = Depends(get_db),
) -> InviteResponse:
    """
    Provision a new org member.

    No email is sent and no credential is returned — there's no email or
    invite-token infrastructure in this codebase yet. The account is
    created inactive with a password nobody knows; a follow-up accept-invite
    flow (set-password token, emailed link) would activate it.
    """
    existing = db.execute(
        select(User).where(User.email == body.email)
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(status_code=409, detail={"error": "email_taken"})

    user = User(
        org_id=_org_uuid(principal),
        email=body.email,
        password_hash=hash_password(secrets.token_urlsafe(32)),
        role=body.role,
        is_active=False,
    )
    db.add(user)
    db.commit()
    return InviteResponse(user_id=str(user.id), email=user.email, role=user.role)


# --- API keys -------------------------------------------------------------------


class ApiKeyOut(BaseModel):
    id: str
    prefix: str
    scopes: list[str]
    last_used_at: datetime | None
    revoked_at: datetime | None


class ApiKeyCreateRequest(BaseModel):
    scopes: list[str] = Field(default_factory=list)


class ApiKeyCreateResponse(BaseModel):
    id: str
    api_key: str  # shown exactly once — never retrievable again
    prefix: str
    scopes: list[str]


@router.get("/api-keys", response_model=list[ApiKeyOut])
def list_api_keys(
    principal: Principal = Depends(current_principal),
    db: Session = Depends(get_db),
) -> list[ApiKeyOut]:
    keys = (
        db.execute(select(ApiKey).where(ApiKey.org_id == _org_uuid(principal)))
        .scalars()
        .all()
    )
    return [
        ApiKeyOut(
            id=str(k.id),
            prefix=k.prefix,
            scopes=list(k.scopes or []),
            last_used_at=k.last_used_at,
            revoked_at=k.revoked_at,
        )
        for k in keys
    ]


@router.post(
    "/api-keys",
    response_model=ApiKeyCreateResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_role("owner", "admin"))],
)
def create_api_key(
    body: ApiKeyCreateRequest,
    principal: Principal = Depends(current_principal),
    db: Session = Depends(get_db),
) -> ApiKeyCreateResponse:
    raw, hashed = generate_api_key()
    key = ApiKey(
        org_id=_org_uuid(principal),
        key_hash=hashed,
        prefix=raw[:12],
        scopes=body.scopes,
    )
    db.add(key)
    db.commit()
    return ApiKeyCreateResponse(
        id=str(key.id), api_key=raw, prefix=key.prefix, scopes=key.scopes
    )


@router.post(
    "/api-keys/{key_id}/revoke",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_role("owner", "admin"))],
)
def revoke_api_key(
    key_id: uuid.UUID,
    principal: Principal = Depends(current_principal),
    db: Session = Depends(get_db),
) -> None:
    key = db.get(ApiKey, key_id)
    if key is None or key.org_id != _org_uuid(principal):
        raise HTTPException(status_code=404, detail={"error": "not_found"})
    if key.revoked_at is None:
        key.revoked_at = datetime.now(timezone.utc)
        db.commit()
    return None
