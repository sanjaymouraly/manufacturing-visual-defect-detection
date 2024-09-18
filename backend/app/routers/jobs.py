"""Batch inspection uploads and background-job management."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import audit, jobs
from app.config import get_settings
from app.db import get_db
from app.models import Inspection, Job, ModelVersion, Product, User
from app.schemas import inspection_out, job_out
from app.security import require_any, require_operator
from app.services.batch import ZipRejected, batch_dir, safe_extract_images
from app.services.storage import safe_filename
from defectml.imaging import ALLOWED_EXTENSIONS

router = APIRouter(prefix="/api", tags=["batch & jobs"])
OPERATOR_JOBS = {"batch_inspection", "report"}


@router.post("/batches", status_code=202)
async def create_batch(files: list[UploadFile] = File(...), product_key: str = Form(...),
                       model_id: int | None = Form(None), batch_code: str = Form("", max_length=128),
                       station: str = Form("", max_length=64), force: bool = Form(False),
                       user: User = Depends(require_operator), db: Session = Depends(get_db)):
    """Upload images and/or ZIP archives; inspection runs in the background (poll ``/api/jobs/{id}``)."""
    s = get_settings()
    product = db.get(Product, product_key)
    if product is None:
        raise HTTPException(422, f"Unknown product '{product_key}'")
    if model_id is not None:
        m = db.get(ModelVersion, model_id)
        if m is None or m.product_key != product_key:
            raise HTTPException(422, "Model does not belong to this product")
    elif product.active_model_id is None:
        raise HTTPException(409, "Product has no active model; activate one in the Model Lab")
    job = jobs.enqueue(db, "batch_inspection", {}, user, max_attempts=3)
    d = batch_dir(job.id)
    saved, skipped, idx = 0, [], 0
    try:
        for f in files:
            name = safe_filename(f.filename or "file")
            ext = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
            if ext == ".zip":
                limit = s.max_zip_mb * 1024 * 1024
                tmp = d / f"_upload_{idx}.zip"
                size = 0
                with open(tmp, "wb") as out:
                    while chunk := await f.read(1 << 20):
                        size += len(chunk)
                        if size > limit:
                            raise HTTPException(413, f"ZIP exceeds {s.max_zip_mb} MB")
                        out.write(chunk)
                got, sk = safe_extract_images(tmp, d, start_index=idx)
                tmp.unlink(missing_ok=True)
                saved += len(got)
                idx += len(got)
                skipped += sk
            elif ext in ALLOWED_EXTENSIONS:
                data = await f.read(s.max_upload_mb * 1024 * 1024 + 1)
                if len(data) > s.max_upload_mb * 1024 * 1024:
                    skipped.append(f"{name}: exceeds {s.max_upload_mb} MB")
                    continue
                (d / f"{idx:05d}_{name}").write_bytes(data)
                saved += 1
                idx += 1
            else:
                skipped.append(f"{name}: unsupported file type")
            if saved > s.max_zip_files:
                raise HTTPException(413, f"Batch exceeds {s.max_zip_files} images")
    except ZipRejected as exc:
        db.rollback()
        import shutil

        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(422, str(exc)) from exc
    except HTTPException:
        db.rollback()
        import shutil

        shutil.rmtree(d, ignore_errors=True)
        raise
    if saved == 0:
        db.rollback()
        import shutil

        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(422, "No valid images found in upload" + (f": {skipped[:3]}" if skipped else ""))
    job.params = {"dir": str(d), "product_key": product_key, "model_id": model_id, "batch_code": batch_code,
                  "station": station, "force": force, "n_files": saved, "skipped": skipped[:100]}
    audit.record(db, "batch.created", user, "job", job.id, {"files": saved, "skipped": len(skipped),
                                                            "product": product_key, "batch_code": batch_code})
    db.commit()
    return job_out(job)


@router.get("/batches/{job_id}/results")
def batch_results(job_id: str, _: User = Depends(require_any), db: Session = Depends(get_db)):
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    rows = db.execute(select(Inspection).where(Inspection.job_id == job_id).order_by(Inspection.filename)).scalars()
    return {"job": job_out(job), "items": [inspection_out(r) for r in rows]}


@router.get("/jobs")
def list_jobs(type: str | None = None, status: str | None = None, limit: int = 50,
              _: User = Depends(require_any), db: Session = Depends(get_db)):
    q = select(Job).order_by(Job.created_at.desc())
    if type:
        q = q.where(Job.type.in_(type.split(",")))
    if status:
        q = q.where(Job.status.in_(status.split(",")))
    return [job_out(j) for j in db.execute(q.limit(min(limit, 500))).scalars()]


def _job(db: Session, job_id: str) -> Job:
    j = db.get(Job, job_id)
    if j is None:
        raise HTTPException(404, "Job not found")
    return j


def _check_perm(user: User, job: Job) -> None:
    if user.role != "admin" and (job.type not in OPERATOR_JOBS or job.created_by_id != user.id):
        raise HTTPException(403, "Not allowed to manage this job")


@router.get("/jobs/{job_id}")
def get_job(job_id: str, _: User = Depends(require_any), db: Session = Depends(get_db)):
    return job_out(_job(db, job_id))


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, user: User = Depends(require_operator), db: Session = Depends(get_db)):
    j = _job(db, job_id)
    _check_perm(user, j)
    if j.status not in ("queued", "running"):
        raise HTTPException(409, f"Job is already {j.status}")
    jobs.request_cancel(db, j)
    audit.record(db, "job.cancel_requested", user, "job", j.id, {"type": j.type})
    db.commit()
    return job_out(j)


@router.post("/jobs/{job_id}/retry")
def retry_job(job_id: str, user: User = Depends(require_operator), db: Session = Depends(get_db)):
    j = _job(db, job_id)
    _check_perm(user, j)
    try:
        jobs.retry(db, j)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    audit.record(db, "job.retried", user, "job", j.id, {"type": j.type})
    db.commit()
    return job_out(j)
