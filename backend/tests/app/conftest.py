"""API test fixtures.

The environment is configured *before* ``app.main`` is imported (it builds the app at import time), using
an isolated temporary var/ directory, SQLite, no background worker (jobs run synchronously via
``run_pending_sync``) and a tiny synthetic demo trained with the dependency-free PixelStats baseline, so the
suite needs neither network access nor pretrained weights.
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_TMP = Path(tempfile.mkdtemp(prefix="defectlens-test-"))
os.environ.update({
    "DL_ENV": "test",
    "DL_VAR_DIR": str(_TMP / "var"),
    "DL_DATABASE_URL": f"sqlite:///{_TMP / 'test.db'}",
    "DL_SECRET_KEY": "test-secret-key-0123456789abcdef0123456789abcdef",
    "DL_ADMIN_USERNAME": "admin",
    "DL_ADMIN_PASSWORD": "Admin-test-pass-2026",
    "DL_WORKER_MODE": "off",
    "DL_WARMUP_MODELS": "false",
    "DL_SERVE_FRONTEND": "false",
    "DL_LOGIN_RATE_LIMIT_PER_MIN": "1000",
    "DL_RATE_LIMIT_PER_MIN": "100000",
    "DL_MAX_UPLOAD_MB": "2",
    "DL_DEVICE": "cpu",
})

from app.config import get_settings  # noqa: E402

get_settings.cache_clear()

from fastapi.testclient import TestClient  # noqa: E402

from app import jobs  # noqa: E402
from app.main import create_app  # noqa: E402

ADMIN = ("admin", "Admin-test-pass-2026")
CSRF = {"X-Requested-With": "defectlens"}


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def app():
    a = create_app()
    with TestClient(a) as c:  # runs the lifespan: migrations + admin seed
        from app.services.bootstrap import bootstrap_demo

        bootstrap_demo(algorithms=("pixelstats",), activate="pixelstats", n_normal=40, n_anomaly=16)
        c.get("/api/health")
    return a


def _client(app, username: str, password: str) -> TestClient:
    c = TestClient(app, headers=CSRF)
    r = c.post("/api/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return c


@pytest.fixture(scope="session")
def admin(app) -> TestClient:
    return _client(app, *ADMIN)


@pytest.fixture(scope="session")
def make_user(app, admin):
    def make(username: str, role: str, password: str = "User-test-pass-2026") -> TestClient:
        r = admin.post("/api/users", json={"username": username, "password": password, "role": role})
        assert r.status_code in (201, 409), r.text
        return _client(app, username, password)
    return make


@pytest.fixture(scope="session")
def inspector(make_user) -> TestClient:
    return make_user("insp1", "inspector")


@pytest.fixture(scope="session")
def viewer(make_user) -> TestClient:
    return make_user("view1", "viewer")


@pytest.fixture(scope="session")
def demo_images():
    """(normal_bytes, anomalous_bytes) from the synthetic demo test split."""
    from defectml.data.manifest import Dataset

    ds = Dataset(get_settings().datasets_dir / "demo-pcb")
    test = ds.split("test")
    n = next(s for s in test if not s.is_anomaly)
    a = next(s for s in test if s.is_anomaly)
    return ds.abspath(n.path).read_bytes(), ds.abspath(a.path).read_bytes()


def run_jobs() -> int:
    return jobs.run_pending_sync()
