"""Authentication and user management."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import audit
from app.config import get_settings
from app.db import get_db
from app.models import User, utcnow
from app.ratelimit import LIMITER, client_ip
from app.schemas import LoginIn, PasswordChange, UserCreate, UserUpdate, user_out
from app.security import (
    COOKIE_NAME,
    create_token,
    get_current_user,
    hash_password,
    require_admin,
    validate_password_strength,
    verify_password,
)

router = APIRouter(prefix="/api", tags=["auth"])


def _authenticate(db: Session, request: Request, username: str, password: str) -> User:
    if not LIMITER.allow(f"login:{client_ip(request)}", get_settings().login_rate_limit_per_min):
        raise HTTPException(429, "Too many login attempts; try again in a minute")
    user = db.execute(select(User).where(User.username == username)).scalar_one_or_none()
    if user is None or not user.is_active or not verify_password(password, user.password_hash):
        audit.record(db, "auth.login_failed", None, "user", username[:64], {"ip": client_ip(request)})
        db.commit()
        raise HTTPException(401, "Invalid username or password")
    user.last_login_at = utcnow()
    audit.record(db, "auth.login", user, "user", user.id, {"ip": client_ip(request)})
    db.commit()
    return user


def _set_cookie(response: Response, token: str) -> None:
    s = get_settings()
    response.set_cookie(COOKIE_NAME, token, httponly=True, secure=s.cookie_secure, samesite="strict",
                        max_age=s.access_token_minutes * 60, path="/")


@router.post("/auth/login")
def login(body: LoginIn, request: Request, response: Response, db: Session = Depends(get_db)):
    """Browser login: sets an httpOnly, SameSite=Strict session cookie."""
    user = _authenticate(db, request, body.username, body.password)
    _set_cookie(response, create_token(user))
    return {"user": user_out(user)}


@router.post("/auth/token")
def token(request: Request, form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """OAuth2 password flow for API / MES clients. Use as ``Authorization: Bearer <token>``."""
    user = _authenticate(db, request, form.username, form.password)
    return {"access_token": create_token(user), "token_type": "bearer"}


@router.post("/auth/logout")
def logout(response: Response, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Ends the session server-side: all tokens issued to this user so far become invalid."""
    u = db.get(User, user.id)
    assert u is not None  # authenticated in this request
    u.token_version = (u.token_version or 0) + 1
    audit.record(db, "auth.logout", u, "user", u.id)
    db.commit()
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"ok": True}


@router.get("/auth/me")
def me(user: User = Depends(get_current_user)):
    return user_out(user)


@router.post("/auth/password")
def change_password(body: PasswordChange, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not verify_password(body.current_password, user.password_hash):
        raise HTTPException(400, "Current password is incorrect")
    validate_password_strength(body.new_password)
    u = db.get(User, user.id)
    assert u is not None  # authenticated in this request
    if body.new_password == body.current_password:
        raise HTTPException(400, "The new password must differ from the current one")
    u.password_hash = hash_password(body.new_password)
    u.password_change_required = False
    u.token_version = (u.token_version or 0) + 1
    audit.record(db, "auth.password_changed", u, "user", u.id)
    db.commit()
    # keep the current browser session alive with a fresh token; other sessions are revoked
    out = JSONResponse({"ok": True})
    _set_cookie(out, create_token(u))
    return out


@router.get("/users")
def list_users(_: User = Depends(require_admin), db: Session = Depends(get_db)):
    return [user_out(u) for u in db.execute(select(User).order_by(User.id)).scalars()]


@router.post("/users", status_code=201)
def create_user(body: UserCreate, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    validate_password_strength(body.password)
    if db.execute(select(User).where(User.username == body.username)).scalar_one_or_none():
        raise HTTPException(409, "Username already exists")
    u = User(username=body.username, full_name=body.full_name, role=body.role, password_hash=hash_password(body.password),
             password_change_required=True)  # initial password was chosen by an admin
    db.add(u)
    db.flush()
    audit.record(db, "user.created", admin, "user", u.id, {"username": u.username, "role": u.role})
    db.commit()
    return user_out(u)


@router.patch("/users/{user_id}")
def update_user(user_id: int, body: UserUpdate, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    u = db.get(User, user_id)
    if u is None:
        raise HTTPException(404, "User not found")
    changes = body.model_dump(exclude_none=True)
    if u.id == admin.id and (changes.get("role", "admin") != "admin" or changes.get("is_active") is False):
        raise HTTPException(409, "You cannot demote or deactivate your own account")
    if "password" in changes:
        validate_password_strength(changes["password"])
        u.password_hash = hash_password(changes.pop("password"))
        u.password_change_required = u.id != admin.id  # temporary password set by an admin must be changed
        changes["password"] = "(changed)"  # noqa: S105 - audit marker, not a secret
    for k in ("full_name", "role", "is_active"):
        if k in changes:
            setattr(u, k, changes[k])
    if {"password", "role", "is_active"} & changes.keys():
        u.token_version = (u.token_version or 0) + 1
    audit.record(db, "user.updated", admin, "user", u.id, changes)
    db.commit()
    return user_out(u)
