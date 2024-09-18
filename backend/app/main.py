"""FastAPI application factory."""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import jobs
from app.config import get_settings
from app.db import init_engine
from app.logging_setup import setup_logging
from app.ratelimit import RequestMiddleware
from app.routers import analytics, auth, datasets, health, inspections, models_api, reviews, settings_api
from app.routers import jobs as jobs_router
from app.services import batch, datasets_service, ml_jobs, reports, retention  # noqa: F401 - register job handlers
from app.services.bootstrap import run_migrations, seed_users, sync_registry

log = logging.getLogger("defectlens")
API_DESCRIPTION = """
**DefectLens** - local-first visual defect inspection API.

* Authenticate with `POST /api/auth/token` (OAuth2 password flow) and send `Authorization: Bearer <token>`,
  or use the browser session cookie (`POST /api/auth/login`; unsafe methods then need `X-Requested-With: defectlens`).
* `POST /api/inspections` inspects one image and returns a persisted, traceable result. Use `batch_code`,
  `external_ref` (MES unit/serial id), `station` and an `Idempotency-Key` header for MES integration.
* Raw anomaly `score` values are unitless and are **not** probabilities. `calibrated_probability` is only present
  when a Platt calibration was fitted on validation data.
"""


def _warmup() -> None:
    """Load active models in the background so the first inspection is not slowed by model loading."""
    from sqlalchemy import select

    from app.db import SessionLocal
    from app.models import ModelVersion, Product
    from app.services.registry import inspector_cache

    try:
        with SessionLocal() as db:
            ids = [i for p in db.execute(select(Product)).scalars() for i in (p.active_model_id, p.secondary_model_id) if i]
            for mid in ids:
                m = db.get(ModelVersion, mid)
                if m is not None:
                    import numpy as np

                    inspector_cache().get(m).model.predict_batch([np.full((256, 256, 3), 128, np.uint8)])
        log.info("warmed up %d model(s)", len(ids))
    except Exception:  # warm-up is best effort; inspections load lazily anyway
        log.exception("model warm-up failed")


def create_app() -> FastAPI:
    s = get_settings()
    setup_logging(s.log_format, s.log_level)
    init_engine(s.database_url)
    worker: jobs.Worker | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal worker
        if s.auto_migrate:
            run_migrations()
        seed_users()
        sync_registry()
        if s.worker_mode == "thread":
            jobs.recover_stale(s.job_stale_seconds)
            worker = jobs.Worker(s.worker_poll_seconds)
            worker.start()
        if s.warmup_models:
            threading.Thread(target=_warmup, name="model-warmup", daemon=True).start()
        log.info("DefectLens started (env=%s, db=%s, worker=%s)", s.env, s.database_url.split("://")[0], s.worker_mode)
        yield
        if worker is not None:
            worker.stop()

    app = FastAPI(title="DefectLens API", version="1.0.0", description=API_DESCRIPTION, lifespan=lifespan,
                  docs_url="/api/docs", redoc_url="/api/redoc", openapi_url="/api/openapi.json")
    app.add_middleware(RequestMiddleware, limit_per_min=s.rate_limit_per_min)
    if s.cors_origins and not s.is_production:
        app.add_middleware(CORSMiddleware, allow_origins=s.cors_origins, allow_credentials=True,
                           allow_methods=["*"], allow_headers=["*"])
    for r in (health, auth, inspections, reviews, jobs_router, models_api, datasets, analytics, settings_api):
        app.include_router(r.router)

    @app.exception_handler(ValueError)
    async def value_error(_: Request, exc: ValueError):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    dist = Path(s.frontend_dist)
    if s.serve_frontend and (dist / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        async def spa(path: str):
            if path.startswith("api/"):
                return JSONResponse({"detail": "Not found"}, status_code=404)
            f = (dist / path).resolve()
            if path and f.is_file() and dist.resolve() in f.parents:
                return FileResponse(f)
            return FileResponse(dist / "index.html", headers={"Cache-Control": "no-cache"})

    return app


app = create_app()
