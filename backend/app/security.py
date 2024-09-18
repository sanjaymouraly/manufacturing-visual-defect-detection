"""Authentication (Argon2 password hashing, signed JWT sessions) and role-based permissions."""

from __future__ import annotations

import base64
import hashlib
from datetime import UTC, datetime, timedelta

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from cryptography.fernet import Fernet, InvalidToken
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.models import User

COOKIE_NAME = "dl_session"
CSRF_HEADER = "x-requested-with"
ROLES = ("admin", "inspector", "viewer")
_ph = PasswordHasher()


def hash_password(pw: str) -> str:
    return _ph.hash(pw)


def verify_password(pw: str, hashed: str) -> bool:
    try:
        return _ph.verify(hashed, pw)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def validate_password_strength(pw: str) -> None:
    if len(pw) < 8:
        raise HTTPException(422, "Password must be at least 8 characters")
    if pw.lower() in {"password", "12345678", "admin123", "change-me-now"}:
        raise HTTPException(422, "Password is too common")


def create_token(user: User) -> str:
    s = get_settings()
    now = datetime.now(UTC)
    payload = {"sub": str(user.id), "role": user.role, "tv": user.token_version or 0, "iat": now,
               "exp": now + timedelta(minutes=s.access_token_minutes)}
    return jwt.encode(payload, s.secret_key, algorithm="HS256")


def decode_token(token: str) -> dict:
    return jwt.decode(token, get_settings().secret_key, algorithms=["HS256"])


def _fernet() -> Fernet:
    key = hashlib.sha256(("defectlens-secrets:" + get_settings().secret_key).encode()).digest()
    return Fernet(base64.urlsafe_b64encode(key))


def encrypt_secret(value: str) -> str:
    return _fernet().encrypt(value.encode()).decode() if value else ""


def decrypt_secret(value: str) -> str:
    if not value:
        return ""
    try:
        return _fernet().decrypt(value.encode()).decode()
    except InvalidToken:
        return ""


def mask_secret(value: str) -> str:
    if not value:
        return ""
    return ("*" * 6) + value[-4:] if len(value) > 8 else "*" * len(value)


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    token = None
    auth = request.headers.get("authorization", "")
    via_cookie = False
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
    else:
        token = request.cookies.get(COOKIE_NAME)
        via_cookie = token is not None
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")
    # CSRF mitigation for cookie sessions: unsafe methods must carry a custom header, which
    # cross-site forms cannot set without a CORS preflight.
    if via_cookie and request.method not in ("GET", "HEAD", "OPTIONS") and request.headers.get(CSRF_HEADER) != "defectlens":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Missing CSRF header")
    try:
        payload = decode_token(token)
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Session expired") from exc
    except jwt.PyJWTError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token") from exc
    user = db.get(User, int(payload["sub"]))
    if user is None or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User inactive or not found")
    if payload.get("tv", 0) != (user.token_version or 0):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Session revoked - please sign in again")
    request.state.user = user
    return user


def require_roles(*roles: str):
    def dep(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"Requires role: {', '.join(roles)}")
        return user

    return dep


require_admin = require_roles("admin")
require_operator = require_roles("admin", "inspector")
require_any = require_roles("admin", "inspector", "viewer")
