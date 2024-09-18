"""Dataset management: registry, statistics, quality reports, previews, custom import, feedback versions."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import audit, jobs
from app.config import get_settings
from app.db import get_db
from app.models import DatasetRecord, User
from app.schemas import job_out
from app.security import require_admin, require_any
from app.services.datasets_service import register_dataset, sync_datasets
from defectml.data.manifest import Dataset

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


def ds_out(r: DatasetRecord) -> dict:
    info = r.info or {}
    q = r.quality_report or {}
    return {"id": r.id, "name": r.name, "product_key": r.product_key, "kind": r.kind, "version": r.version,
            "sample_count": r.sample_count, "manifest_sha256": r.manifest_sha256, "created_at": r.created_at.isoformat(),
            "notes": r.notes, "license": info.get("license"), "license_url": info.get("license_url"),
            "source": info.get("source"), "source_url": info.get("source_url"), "citation": info.get("citation"),
            "synthetic": info.get("synthetic"), "defect_types_annotated": info.get("defect_types_annotated"),
            "masks_available": info.get("masks_available"), "split_policy": info.get("split_policy"),
            "preprocessing": info.get("preprocessing"), "description": info.get("description"),
            "labels": q.get("labels"), "splits": q.get("splits"), "defect_types": q.get("defect_types"),
            "valid": q.get("valid"), "warnings": q.get("warnings", [])}


def _rec(db: Session, dataset_id: int) -> DatasetRecord:
    r = db.get(DatasetRecord, dataset_id)
    if r is None:
        raise HTTPException(404, "Dataset not found")
    return r


@router.get("")
def list_datasets(_: User = Depends(require_any), db: Session = Depends(get_db)):
    return [ds_out(r) for r in db.execute(select(DatasetRecord).order_by(DatasetRecord.name)).scalars()]


@router.post("/sync")
def sync(user: User = Depends(require_admin), db: Session = Depends(get_db)):
    recs = sync_datasets(db)
    audit.record(db, "dataset.sync", user, "dataset", "", {"count": len(recs)})
    db.commit()
    return {"count": len(recs)}


@router.get("/{dataset_id}")
def get_dataset(dataset_id: int, _: User = Depends(require_any), db: Session = Depends(get_db)):
    r = _rec(db, dataset_id)
    return {**ds_out(r), "quality_report": r.quality_report}


@router.post("/{dataset_id}/validate")
def validate(dataset_id: int, user: User = Depends(require_admin), db: Session = Depends(get_db)):
    r = _rec(db, dataset_id)
    register_dataset(db, Path(r.path), user, validate=True)
    db.commit()
    return {**ds_out(r), "quality_report": r.quality_report}


@router.get("/{dataset_id}/preview.jpg")
def preview(dataset_id: int, _: User = Depends(require_any), db: Session = Depends(get_db)):
    r = _rec(db, dataset_id)
    p = Path(r.path) / "preview.jpg"
    if not p.exists():
        raise HTTPException(404, "Preview not generated; run validation")
    return FileResponse(p, media_type="image/jpeg")


@router.get("/{dataset_id}/samples")
def samples(dataset_id: int, split: str | None = None, label: str | None = None, offset: int = 0, limit: int = 48,
            _: User = Depends(require_any), db: Session = Depends(get_db)):
    ds = Dataset(_rec(db, dataset_id).path)
    rows = [s for s in ds.samples if (not split or s.split == split) and (not label or s.label == label)]
    return {"total": len(rows), "items": [{"index": ds.samples.index(s), "path": s.path, "label": s.label,
                                           "defect_type": s.defect_type, "split": s.split, "has_mask": bool(s.mask),
                                           "group": s.group} for s in rows[offset:offset + min(limit, 200)]]}


@router.get("/{dataset_id}/samples/{index}/{kind}")
def sample_file(dataset_id: int, index: int, kind: str, _: User = Depends(require_any), db: Session = Depends(get_db)):
    ds = Dataset(_rec(db, dataset_id).path)
    if not 0 <= index < len(ds.samples) or kind not in ("image", "mask"):
        raise HTTPException(404, "Not found")
    s = ds.samples[index]
    rel = s.path if kind == "image" else s.mask
    if not rel:
        raise HTTPException(404, "No mask for this sample")
    return FileResponse(ds.abspath(rel))


@router.post("/import", status_code=202)
async def import_dataset(file: UploadFile = File(...), name: str = Form(..., pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$"),
                         product_key: str = Form(..., pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$"),
                         license: str = Form("user-provided", max_length=128),
                         user: User = Depends(require_admin), db: Session = Depends(get_db)):
    """Upload a ZIP with a custom dataset (MVTec-style or normal/ + anomaly/ folders, see docs/DATASETS.md)."""
    s = get_settings()
    if db.execute(select(DatasetRecord).where(DatasetRecord.name == name)).scalar_one_or_none():
        raise HTTPException(409, "Dataset name already exists")
    job = jobs.enqueue(db, "import_dataset", {}, user, max_attempts=1)
    d = s.tmp_dir / "imports" / job.id
    d.mkdir(parents=True, exist_ok=True)
    zpath = d / "upload.zip"
    size, limit = 0, s.max_zip_mb * 1024 * 1024
    with open(zpath, "wb") as out:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > limit:
                db.rollback()
                raise HTTPException(413, f"Archive exceeds {s.max_zip_mb} MB")
            out.write(chunk)
    job.params = {"zip_path": str(zpath), "name": name, "product_key": product_key, "license": license}
    audit.record(db, "dataset.import_requested", user, "job", job.id, {"name": name, "bytes": size})
    db.commit()
    return job_out(job)


class FeedbackBuildIn(BaseModel):
    base_dataset_id: int
    seed: int = Field(42, ge=0)


@router.post("/feedback-version", status_code=202)
def build_feedback_version(body: FeedbackBuildIn, user: User = Depends(require_admin), db: Session = Depends(get_db)):
    _rec(db, body.base_dataset_id)
    job = jobs.enqueue(db, "build_feedback_dataset", body.model_dump(), user, max_attempts=1)
    db.commit()
    return job_out(job)
