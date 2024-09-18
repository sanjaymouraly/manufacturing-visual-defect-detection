"""Health, readiness and metrics endpoints (unauthenticated, no sensitive data)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, PlainTextResponse
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.metrics import METRICS
from app.models import Inspection, Job, Product

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/health/ready")
def ready(db: Session = Depends(get_db)):
    checks: dict[str, str | int] = {}
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = f"error: {exc.__class__.__name__}"
    s = get_settings()
    try:  # actually write: an existing but read-only volume must not report ok
        probe = s.data_dir / ".ready-probe"
        probe.write_bytes(b"ok")
        probe.unlink()
        checks["storage_writable"] = "ok"
    except OSError as exc:
        checks["storage_writable"] = f"error: {exc.__class__.__name__}"
    n_active = db.execute(select(func.count()).select_from(Product).where(Product.active_model_id.is_not(None))).scalar() \
        if checks["database"] == "ok" else 0
    checks["active_models"] = int(n_active or 0)
    ok = checks["database"] == "ok" and checks["storage_writable"] == "ok"
    return JSONResponse({"status": "ready" if ok else "not_ready", "checks": checks}, status_code=200 if ok else 503)


@router.get("/metrics", response_class=PlainTextResponse)
def metrics(db: Session = Depends(get_db)):
    gauges = {
        "jobs_queued": db.execute(select(func.count()).select_from(Job).where(Job.status == "queued")).scalar() or 0,
        "jobs_running": db.execute(select(func.count()).select_from(Job).where(Job.status == "running")).scalar() or 0,
        "reviews_pending": db.execute(select(func.count()).select_from(Inspection)
                                      .where(Inspection.review_status == "pending")).scalar() or 0,
    }
    return METRICS.render({k: float(v) for k, v in gauges.items()})
