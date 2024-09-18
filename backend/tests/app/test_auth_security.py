"""Authentication, sessions, CSRF, RBAC, rate limiting and audit-trail integrity."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from .conftest import ADMIN, CSRF


def test_health_is_public(app):
    c = TestClient(app)
    assert c.get("/api/health").json()["status"] == "ok"
    ready = c.get("/api/health/ready")
    assert ready.status_code == 200 and ready.json()["checks"]["database"] == "ok"


def test_protected_routes_require_auth(app):
    c = TestClient(app)
    for path in ("/api/inspections", "/api/models", "/api/dashboard", "/api/users", "/api/audit"):
        assert c.get(path).status_code == 401, path


def test_wrong_password_is_rejected_and_audited(app, admin):
    c = TestClient(app, headers=CSRF)
    r = c.post("/api/auth/login", json={"username": "admin", "password": "wrong-password"})
    assert r.status_code == 401
    r = c.post("/api/auth/login", json={"username": "nobody", "password": "whatever-123"})
    assert r.status_code == 401  # same message, no user enumeration
    actions = [e["action"] for e in admin.get("/api/audit", params={"action": "auth.login_failed"}).json()]
    assert actions and set(actions) == {"auth.login_failed"}


def test_session_cookie_is_httponly_strict(app):
    c = TestClient(app, headers=CSRF)
    r = c.post("/api/auth/login", json={"username": ADMIN[0], "password": ADMIN[1]})
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    assert "password_hash" not in r.text


def test_csrf_header_required_for_cookie_mutations(app, make_user):
    c = make_user("csrf_user", "inspector")
    no_header = TestClient(app, cookies=c.cookies)
    r = no_header.post("/api/reviews/1", json={"label": "normal"})
    assert r.status_code == 403
    assert "x-requested-with" in r.text.lower() or "csrf" in r.text.lower()


def test_bearer_token_flow(app):
    c = TestClient(app)
    r = c.post("/api/auth/token", data={"username": ADMIN[0], "password": ADMIN[1]})
    assert r.status_code == 200
    tok = r.json()["access_token"]
    me = c.get("/api/auth/me", headers={"Authorization": f"Bearer {tok}"})
    assert me.status_code == 200 and me.json()["role"] == "admin"
    assert c.get("/api/auth/me", headers={"Authorization": "Bearer not-a-token"}).status_code == 401


def test_logout_revokes_existing_tokens(app, make_user):
    c = make_user("logout_user", "inspector")
    stolen = dict(c.cookies)
    assert c.post("/api/auth/logout").status_code == 200
    replay = TestClient(app, cookies=stolen, headers=CSRF)
    assert replay.get("/api/auth/me").status_code == 401


def test_password_change_revokes_other_sessions(app, make_user):
    a = make_user("pw_user", "inspector", "Pw-user-pass-2026")
    other = TestClient(app, headers=CSRF)
    other.post("/api/auth/login", json={"username": "pw_user", "password": "Pw-user-pass-2026"})
    r = a.post("/api/auth/password", json={"current_password": "Pw-user-pass-2026", "new_password": "Pw-user-pass-2027!"})
    assert r.status_code == 200
    assert a.get("/api/auth/me").status_code == 200  # current session refreshed
    assert other.get("/api/auth/me").status_code == 401  # other session revoked


def test_weak_password_rejected(admin):
    r = admin.post("/api/users", json={"username": "weakling", "password": "password", "role": "viewer"})
    assert r.status_code in (400, 422)


def test_rbac_viewer_and_inspector(viewer, inspector, demo_images):
    normal, _ = demo_images
    # viewer: read-only
    assert viewer.get("/api/inspections").status_code == 200
    r = viewer.post("/api/inspections", files={"file": ("x.png", normal, "image/png")}, data={"product_key": "demo-pcb"})
    assert r.status_code == 403
    # inspector: can inspect but cannot administer
    assert inspector.get("/api/users").status_code == 403
    assert inspector.put("/api/settings/policy", json={"value": {}}).status_code == 403
    assert inspector.post("/api/products/demo-pcb/rollback").status_code == 403
    assert inspector.post("/api/retention/run").status_code == 403


def test_login_rate_limit(app, monkeypatch):
    from app import config
    from app.ratelimit import LIMITER

    s = config.get_settings()
    monkeypatch.setattr(s, "login_rate_limit_per_min", 3)
    LIMITER.reset() if hasattr(LIMITER, "reset") else LIMITER._hits.clear()
    c = TestClient(app, headers=CSRF)
    codes = [c.post("/api/auth/login", json={"username": "x", "password": "y-123456"}).status_code for _ in range(5)]
    assert codes[:3] == [401, 401, 401] and codes[3:] == [429, 429]
    LIMITER._hits.clear()


def test_audit_chain_valid_and_append_only(app, admin):
    r = admin.get("/api/audit/verify").json()
    assert r["valid"] is True and r["checked"] > 0

    from app.db import SessionLocal
    from app.models import AppendOnlyViolation, AuditEvent

    with SessionLocal() as db:
        ev = db.query(AuditEvent).first()
        ev.action = "tampered"
        with pytest.raises(AppendOnlyViolation):
            db.flush()
        db.rollback()
        ev = db.query(AuditEvent).first()
        db.delete(ev)
        with pytest.raises(AppendOnlyViolation):
            db.flush()
        db.rollback()


def test_audit_chain_detects_raw_sql_tampering(app, admin):
    from sqlalchemy import text

    from app.db import SessionLocal

    with SessionLocal() as db:
        row = db.execute(text("SELECT id, action FROM audit_events ORDER BY id LIMIT 1")).one()
        db.execute(text("UPDATE audit_events SET action='forged' WHERE id=:i"), {"i": row.id})
        db.commit()
        try:
            r = admin.get("/api/audit/verify").json()
            assert r["valid"] is False and r["broken_at"] == row.id
        finally:
            db.execute(text("UPDATE audit_events SET action=:a WHERE id=:i"), {"a": row.action, "i": row.id})
            db.commit()
    assert admin.get("/api/audit/verify").json()["valid"] is True


def test_security_headers(app):
    r = TestClient(app).get("/api/health")
    h = {k.lower(): v for k, v in r.headers.items()}
    assert h.get("x-content-type-options") == "nosniff"
    csp = h["content-security-policy"]
    assert "script-src 'self'" in csp and "object-src 'none'" in csp and "unsafe-eval" not in csp
    assert "frame-ancestors" not in csp  # only enforced in production (or when DL_FRAME_ANCESTORS is set)
    assert "camera=(self)" in h["permissions-policy"]
    assert "content-security-policy" not in {k.lower() for k in TestClient(app).get("/docs").headers}


def test_csp_frame_ancestors_in_production(monkeypatch):
    from app import config
    from app.ratelimit import _csp

    s = config.get_settings()
    monkeypatch.setattr(s, "env", "production")
    assert "frame-ancestors 'self'" in _csp()
    monkeypatch.setattr(s, "frame_ancestors", "https://mes.example.com")
    assert "frame-ancestors https://mes.example.com" in _csp()


def test_password_change_required_lifecycle(app, admin):
    from .conftest import CSRF as H

    r = admin.post("/api/users", json={"username": "newbie", "password": "Initial-pass-2026", "role": "inspector"})
    assert r.status_code == 201 and r.json()["password_change_required"] is True  # admin-chosen initial password
    c = TestClient(app, headers=H)
    c.post("/api/auth/login", json={"username": "newbie", "password": "Initial-pass-2026"})
    assert c.get("/api/auth/me").json()["password_change_required"] is True
    same = c.post("/api/auth/password", json={"current_password": "Initial-pass-2026", "new_password": "Initial-pass-2026"})
    assert same.status_code == 400
    ok = c.post("/api/auth/password", json={"current_password": "Initial-pass-2026", "new_password": "My-own-pass-2026"})
    assert ok.status_code == 200
    assert c.get("/api/auth/me").json()["password_change_required"] is False
    # an admin reset sets a temporary password -> change required again
    uid = c.get("/api/auth/me").json()["id"]
    assert admin.patch(f"/api/users/{uid}", json={"password": "Temp-reset-2026"}).json()["password_change_required"] is True


def test_seeded_demo_accounts_flagged(app):
    from app.db import SessionLocal
    from app.models import User

    with SessionLocal() as db:
        demo = {u.username: u.password_change_required for u in db.query(User).filter(User.username.in_(["inspector", "viewer"]))}
    assert demo and all(demo.values())
