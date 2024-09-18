"""Dashboard, analytics, reports and AI summaries over inspection history."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app import jobs
from app.db import get_db
from app.models import User
from app.schemas import AskIn, ReportIn, job_out
from app.security import require_any, require_operator
from app.services import llm
from app.services.analytics import full_analytics, overview
from app.services.reports import report_path

router = APIRouter(prefix="/api", tags=["analytics & reports"])


@router.get("/dashboard")
def dashboard(product_key: str | None = None, days: int = 30, _: User = Depends(require_any),
              db: Session = Depends(get_db)):
    return overview(db, product_key, max(1, min(days, 3650)))


@router.get("/analytics")
def analytics(product_key: str | None = None, days: int = 30, _: User = Depends(require_any),
              db: Session = Depends(get_db)):
    return full_analytics(db, product_key, max(1, min(days, 3650)))


@router.post("/reports", status_code=202)
def create_report(body: ReportIn, user: User = Depends(require_operator), db: Session = Depends(get_db)):
    job = jobs.enqueue(db, "report", body.model_dump(), user, max_attempts=2)
    db.commit()
    return job_out(job)


@router.get("/reports/{name}")
def download_report(name: str, _: User = Depends(require_any)):
    try:
        p = report_path(name)
    except ValueError as exc:
        raise HTTPException(404, "Not found") from exc
    if not p.exists() or p.suffix != ".pdf":
        raise HTTPException(404, "Report not found")
    return FileResponse(p, media_type="application/pdf", filename=p.name)


@router.post("/ai/period-summary")
def period_summary(product_key: str | None = None, days: int = 30, provider_id: int | None = None,
                   _: User = Depends(require_any), db: Session = Depends(get_db)):
    try:
        return llm.summarize_period(db, product_key, days, provider_id)
    except llm.PrivacyBlocked as exc:
        raise HTTPException(403, str(exc)) from exc
    except llm.LLMError as exc:
        raise HTTPException(502, f"AI provider error: {exc}") from exc


@router.post("/ai/ask")
def ask(body: AskIn, _: User = Depends(require_any), db: Session = Depends(get_db)):
    try:
        return llm.answer_question(db, body.question, body.product_key, body.days, body.provider_id)
    except llm.PrivacyBlocked as exc:
        raise HTTPException(403, str(exc)) from exc
    except llm.LLMError as exc:
        raise HTTPException(502, f"AI provider error: {exc}") from exc
