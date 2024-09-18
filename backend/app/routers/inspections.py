"""Inspection API: single-image inspection (UI, camera and MES/API clients), history, exports."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db
from app.models import AuditEvent, Inspection, Review, User
from app.schemas import inspection_out
from app.security import require_any, require_operator
from app.services import llm
from app.services.inspection import InspectionInputError, run_inspection
from app.services.reports import inspection_pdf, to_csv
from app.services.storage import abs_data_path

router = APIRouter(prefix="/api/inspections", tags=["inspections"])


async def read_limited(file: UploadFile, limit_mb: int) -> bytes:
    limit = limit_mb * 1024 * 1024
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"File exceeds {limit_mb} MB limit")
    return data


@router.post("", status_code=201)
async def inspect(
    file: UploadFile = File(...),
    product_key: str = Form(...),
    model_id: int | None = Form(None),
    batch_code: str = Form("", max_length=128),
    camera_id: str = Form("", max_length=64),
    station: str = Form("", max_length=64),
    external_ref: str = Form("", max_length=128),
    source: str = Form("upload", pattern="^(upload|camera|api)$"),
    force: bool = Form(False),
    idempotency_key: str | None = Header(None, alias="Idempotency-Key", max_length=128),
    user: User = Depends(require_operator),
    db: Session = Depends(get_db),
):
    """Inspect one image. Returns the persisted inspection (``duplicate=true`` if an identical image was
    already inspected with the same model within the duplicate window, or the Idempotency-Key was reused)."""
    data = await read_limited(file, get_settings().max_upload_mb)
    try:
        insp, dup = run_inspection(db, data=data, filename=file.filename or "upload", product_key=product_key,
                                   operator=user, model_id=model_id, source=source, batch_code=batch_code,
                                   camera_id=camera_id, station=station, external_ref=external_ref,
                                   idempotency_key=idempotency_key, force=force,
                                   persist_decode_errors=source in ("camera", "api"))
        db.commit()
    except InspectionInputError as exc:
        db.rollback()
        raise HTTPException(422, str(exc)) from exc
    db.refresh(insp)
    return {**inspection_out(insp, detail=True), "duplicate": dup}


def _filtered(q, *, search, product_key, decision, batch_code, camera_id, model_id, review_status, source,
              date_from, date_to, job_id, final_label, reviewer, include_duplicates=True):
    if search:
        like = f"%{search}%"
        q = q.where(or_(Inspection.filename.ilike(like), Inspection.id.ilike(like), Inspection.external_ref.ilike(like),
                        Inspection.batch_code.ilike(like)))
    for col, val in ((Inspection.product_key, product_key), (Inspection.decision, decision),
                     (Inspection.batch_code, batch_code), (Inspection.camera_id, camera_id),
                     (Inspection.model_id, model_id), (Inspection.review_status, review_status),
                     (Inspection.source, source), (Inspection.job_id, job_id), (Inspection.final_label, final_label)):
        if val not in (None, ""):
            q = q.where(col == val)
    if date_from:
        q = q.where(Inspection.created_at >= date_from)
    if date_to:
        q = q.where(Inspection.created_at <= date_to)
    if reviewer:
        sub = select(Review.inspection_id).join(User, Review.reviewer_id == User.id).where(User.username == reviewer)
        q = q.where(Inspection.id.in_(sub))
    if not include_duplicates:
        q = q.where(Inspection.duplicate_of.is_(None))
    return q


def _filter_params(
    q: str | None = Query(None, max_length=128), product_key: str | None = None, decision: str | None = None,
    batch_code: str | None = None, camera_id: str | None = None, model_id: int | None = None,
    review_status: str | None = None, source: str | None = None, date_from: datetime | None = None,
    date_to: datetime | None = None, job_id: str | None = None, final_label: str | None = None,
    reviewer: str | None = None,
) -> dict:
    return dict(search=q, product_key=product_key, decision=decision, batch_code=batch_code, camera_id=camera_id,
                model_id=model_id, review_status=review_status, source=source, date_from=date_from, date_to=date_to,
                job_id=job_id, final_label=final_label, reviewer=reviewer)


@router.get("")
def list_inspections(f: dict = Depends(_filter_params), page: int = Query(1, ge=1),
                     page_size: int = Query(25, ge=1, le=200), sort: str = Query("-created_at"),
                     _: User = Depends(require_any), db: Session = Depends(get_db)):
    base = _filtered(select(Inspection), **f)
    total = db.execute(select(func.count()).select_from(base.subquery())).scalar()
    col = {"created_at": Inspection.created_at, "score": Inspection.normalized_score}.get(sort.lstrip("-"),
                                                                                         Inspection.created_at)
    base = base.order_by(col.desc() if sort.startswith("-") else col.asc())
    rows = db.execute(base.offset((page - 1) * page_size).limit(page_size)).scalars().all()
    return {"items": [inspection_out(r) for r in rows], "total": total, "page": page, "page_size": page_size}


@router.get("/export.csv")
def export_csv(f: dict = Depends(_filter_params), _: User = Depends(require_any), db: Session = Depends(get_db)):
    rows = db.execute(_filtered(select(Inspection), **f).order_by(Inspection.created_at.desc()).limit(100000)).scalars().all()
    return Response(to_csv(list(rows)), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="inspections-{datetime.now():%Y%m%d-%H%M}.csv"'})


@router.get("/facets")
def facets(_: User = Depends(require_any), db: Session = Depends(get_db)):
    def distinct(col):
        return [v for v in db.execute(select(col).distinct().where(col != "").limit(500)).scalars() if v]

    reviewers = db.execute(select(User.username).join(Review, Review.reviewer_id == User.id).distinct()).scalars().all()
    return {"batch_codes": distinct(Inspection.batch_code), "camera_ids": distinct(Inspection.camera_id),
            "products": distinct(Inspection.product_key), "reviewers": reviewers,
            "models": [{"id": i, "name": n} for i, n in db.execute(
                select(Inspection.model_id, Inspection.model_name).distinct().where(Inspection.model_id.is_not(None)))]}


def _get(db: Session, inspection_id: str) -> Inspection:
    i = db.get(Inspection, inspection_id)
    if i is None:
        raise HTTPException(404, "Inspection not found")
    return i


@router.get("/{inspection_id}")
def get_inspection(inspection_id: str, _: User = Depends(require_any), db: Session = Depends(get_db)):
    return inspection_out(_get(db, inspection_id), detail=True)


@router.get("/{inspection_id}/image/{kind}")
def get_image(inspection_id: str, kind: str, _: User = Depends(require_any), db: Session = Depends(get_db)):
    i = _get(db, inspection_id)
    if kind not in ("display", "heatmap", "overlay", "mask", "original"):
        raise HTTPException(404, "Unknown image kind")
    rel = i.image_path if kind == "original" else getattr(i, f"{kind}_path")
    if not rel:
        raise HTTPException(404, "Image not available (not produced or purged by retention policy)")
    p = abs_data_path(rel)
    if not p.exists():
        raise HTTPException(404, "Image file missing")
    media = {".jpg": "image/jpeg", ".png": "image/png"}.get(p.suffix, "application/octet-stream")
    return FileResponse(p, media_type=media, headers={"Cache-Control": "private, max-age=3600"})


@router.get("/{inspection_id}/report.pdf")
def inspection_report(inspection_id: str, _: User = Depends(require_any), db: Session = Depends(get_db)):
    i = _get(db, inspection_id)
    return Response(inspection_pdf(i), media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="inspection-{i.id[:12]}.pdf"'})


@router.post("/{inspection_id}/summary")
def inspection_summary(inspection_id: str, provider_id: int | None = None, _: User = Depends(require_any),
                       db: Session = Depends(get_db)):
    i = _get(db, inspection_id)
    try:
        return llm.summarize_inspection(db, i, provider_id)
    except llm.PrivacyBlocked as exc:
        raise HTTPException(403, str(exc)) from exc
    except llm.LLMError as exc:
        raise HTTPException(502, f"AI provider error: {exc}") from exc


@router.get("/{inspection_id}/audit")
def inspection_audit(inspection_id: str, _: User = Depends(require_any), db: Session = Depends(get_db)):
    rows = db.execute(select(AuditEvent).where(AuditEvent.entity_type == "inspection",
                                               AuditEvent.entity_id == inspection_id).order_by(AuditEvent.id)).scalars()
    return [{"id": e.id, "created_at": e.created_at.isoformat(), "actor": e.actor_name, "action": e.action,
             "details": e.details, "hash": e.hash} for e in rows]
