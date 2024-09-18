"""Batch inspection: safe ZIP extraction and the background batch job."""

from __future__ import annotations

import json
import shutil
import stat
import zipfile
from pathlib import Path

from sqlalchemy import select

from app.config import get_settings
from app.db import SessionLocal
from app.jobs import JobContext, PermanentJobError, handler
from app.models import Inspection, User
from app.services.inspection import InspectionInputError, run_inspection
from app.services.storage import safe_filename
from defectml.imaging import ALLOWED_EXTENSIONS


class ZipRejected(ValueError):
    pass


def safe_extract_images(zip_path: Path, dest: Path, start_index: int = 0) -> tuple[list[Path], list[str]]:
    """Extract image members only, defending against zip-slip, symlinks, bombs and oversize files."""
    s = get_settings()
    max_member = s.max_upload_mb * 1024 * 1024
    max_total = s.max_zip_uncompressed_mb * 1024 * 1024
    saved: list[Path] = []
    skipped: list[str] = []
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise ZipRejected("Not a valid ZIP archive") from exc
    with zf:
        members = zf.infolist()
        files = [m for m in members if not m.is_dir()]
        if len(files) > s.max_zip_files:
            raise ZipRejected(f"ZIP contains {len(files)} files; limit is {s.max_zip_files}")
        total = sum(m.file_size for m in files)
        if total > max_total:
            raise ZipRejected("ZIP uncompressed size exceeds the configured limit")
        idx = start_index
        for m in files:
            name = m.filename.replace("\\", "/")
            base = name.rsplit("/", 1)[-1]
            if name.startswith("__MACOSX/") or base.startswith("."):
                continue
            mode = (m.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                skipped.append(f"{name}: symbolic link skipped")
                continue
            if Path(base).suffix.lower() not in ALLOWED_EXTENSIONS:
                skipped.append(f"{name}: unsupported file type")
                continue
            if m.compress_size and m.file_size / m.compress_size > 200:
                raise ZipRejected(f"Suspicious compression ratio for {name} (possible zip bomb)")
            if m.file_size > max_member:
                skipped.append(f"{name}: exceeds {s.max_upload_mb} MB")
                continue
            with zf.open(m) as src:
                data = src.read(max_member + 1)
            if len(data) > max_member:
                skipped.append(f"{name}: exceeds {s.max_upload_mb} MB")
                continue
            out = dest / f"{idx:05d}_{safe_filename(base)}"
            out.write_bytes(data)
            saved.append(out)
            idx += 1
    return saved, skipped


def batch_dir(job_id: str) -> Path:
    d = get_settings().tmp_dir / "batches" / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


@handler("batch_inspection")
def batch_job(ctx: JobContext) -> dict:
    p = ctx.params
    d = Path(p["dir"])
    if not d.exists():
        raise PermanentJobError("Batch files are missing (already processed or deleted)")
    files = sorted(x for x in d.iterdir() if x.is_file() and x.name != "manifest.json")
    total = len(files)
    counts = {"PASS": 0, "FAIL": 0, "REVIEW": 0, "duplicates": 0, "errors": 0}
    with SessionLocal() as db:
        done_names = set(db.execute(select(Inspection.filename).where(Inspection.job_id == ctx.job_id)).scalars())
        user = db.get(User, ctx.user_id) if ctx.user_id else None
        for i, f in enumerate(files):
            ctx.check_cancel()
            fname = f.name.split("_", 1)[1] if "_" in f.name else f.name
            if safe_filename(fname) in done_names:  # resume after retry/crash
                continue
            try:
                insp, dup = run_inspection(db, data=f.read_bytes(), filename=fname, product_key=p["product_key"],
                                           operator=user, model_id=p.get("model_id"), source="batch",
                                           batch_code=p.get("batch_code", ""), station=p.get("station", ""),
                                           job_id=ctx.job_id, force=bool(p.get("force")), persist_decode_errors=True)
                if dup:
                    counts["duplicates"] += 1
                    # keep traceability: record a lightweight duplicate entry for this job
                    clone = Inspection(**{c.name: getattr(insp, c.name) for c in Inspection.__table__.columns
                                          if c.name not in ("id", "job_id", "idempotency_key", "created_at",
                                                            "duplicate_of", "filename", "review_status",
                                                            "needs_review")})
                    clone.job_id, clone.duplicate_of, clone.filename = ctx.job_id, insp.id, safe_filename(fname)
                    clone.review_status, clone.needs_review = "none", False
                    clone.reasons = [*list(insp.reasons or []), f"Duplicate of inspection {insp.id}"]
                    db.add(clone)
                else:
                    counts[insp.decision] += 1
                    if insp.status == "error":
                        counts["errors"] += 1
                db.commit()
            except InspectionInputError as exc:
                db.rollback()
                raise PermanentJobError(str(exc)) from exc
            except Exception:
                db.rollback()
                raise
            ctx.progress((i + 1) / max(total, 1), f"Inspected {i + 1}/{total}")
    shutil.rmtree(d, ignore_errors=True)
    return {"total": total, **counts, "skipped": p.get("skipped", [])}


def write_manifest(d: Path, info: dict) -> None:
    (d / "manifest.json").write_text(json.dumps(info), encoding="utf-8")
