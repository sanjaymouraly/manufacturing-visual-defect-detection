"""Data retention (configurable purge of images / records) and backup / restore."""

from __future__ import annotations

import json
import shutil
import sqlite3
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import delete, select

from app import audit
from app.config import get_settings
from app.db import SessionLocal
from app.jobs import JobContext, handler
from app.models import FeedbackSample, Inspection, Job, Review, utcnow
from app.services.settings_service import get_setting
from app.services.storage import delete_inspection_files


def apply_retention(dry_run: bool = False) -> dict:
    with SessionLocal() as db:
        cfg = get_setting(db, "retention")
        res = {"images_purged": 0, "inspections_deleted": 0, "jobs_deleted": 0, "dry_run": dry_run}
        now = utcnow()
        if cfg["image_days"] and cfg["image_days"] > 0:
            cutoff = now - timedelta(days=int(cfg["image_days"]))
            q = select(Inspection).where(Inspection.created_at < cutoff, Inspection.images_purged.is_(False))
            for i in db.execute(q).scalars():
                if cfg.get("keep_reviewed_images") and i.final_label:
                    continue
                if i.review_status == "pending":
                    continue  # never purge evidence awaiting review
                res["images_purged"] += 1
                if not dry_run:
                    if i.image_path and i.duplicate_of is None:
                        delete_inspection_files(i.image_path.rsplit("/", 1)[0])
                    i.images_purged = True
                    i.image_path = i.display_path = i.heatmap_path = i.overlay_path = i.mask_path = ""
        if cfg["inspection_days"] and cfg["inspection_days"] > 0:
            cutoff = now - timedelta(days=int(cfg["inspection_days"]))
            old = db.execute(select(Inspection).where(Inspection.created_at < cutoff,
                                                      Inspection.review_status != "pending")).scalars().all()
            # reviewed records referenced by feedback/reviews are kept for traceability
            keep = set(db.execute(select(Review.inspection_id)).scalars()) | set(
                db.execute(select(FeedbackSample.inspection_id)).scalars())
            for i in old:
                if i.id in keep:
                    continue
                res["inspections_deleted"] += 1
                if not dry_run:
                    if i.image_path and i.duplicate_of is None:
                        delete_inspection_files(i.image_path.rsplit("/", 1)[0])
                    db.delete(i)
        if cfg["job_days"] and cfg["job_days"] > 0:
            cutoff = now - timedelta(days=int(cfg["job_days"]))
            jq = select(Job).where(Job.finished_at < cutoff, Job.status.in_(["succeeded", "failed", "cancelled"]))
            jobs = db.execute(jq).scalars().all()
            referenced = set(db.execute(select(Inspection.job_id).where(Inspection.job_id.is_not(None))).scalars())
            res["jobs_deleted"] = sum(1 for j in jobs if j.id not in referenced)
            if not dry_run:
                ids = [j.id for j in jobs if j.id not in referenced]
                if ids:
                    db.execute(delete(Job).where(Job.id.in_(ids)))
        if not dry_run:
            audit.record(db, "retention.applied", None, "system", "retention", res)
            db.commit()
        return res


@handler("retention")
def retention_job(ctx: JobContext) -> dict:
    return apply_retention(bool(ctx.params.get("dry_run")))


def backup(out: Path) -> Path:
    """Consistent backup: SQLite online backup API + data files + models registry metadata."""
    s = get_settings()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = s.tmp_dir / f"backup-{datetime.now(UTC):%Y%m%d%H%M%S}"
    tmp.mkdir(parents=True)
    manifest = {"created_at": datetime.now(UTC).isoformat(), "database_url": s.database_url.split("@")[-1]}
    if s.database_url.startswith("sqlite:///"):
        src_path = s.database_url.replace("sqlite:///", "")
        src = sqlite3.connect(src_path)
        dst = sqlite3.connect(tmp / "defectlens.db")
        with dst:
            src.backup(dst)
        src.close()
        dst.close()
    else:
        manifest["note"] = "PostgreSQL detected: back up the database with pg_dump (see docs/OPERATIONS.md)."
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in tmp.rglob("*"):
            zf.write(f, f.relative_to(tmp).as_posix())
        for base, prefix in ((s.data_dir, "data"), (s.models_dir, "models")):
            for f in base.rglob("*"):
                if f.is_file():
                    zf.write(f, f"{prefix}/{f.relative_to(base).as_posix()}")
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def restore(archive: Path) -> dict:
    """Restore a backup created by :func:`backup` (stop the server first)."""
    s = get_settings()
    restored = {"database": False, "files": 0}
    with zipfile.ZipFile(archive) as zf:
        for m in zf.infolist():
            name = m.filename
            if name == "defectlens.db" and s.database_url.startswith("sqlite:///"):
                target = Path(s.database_url.replace("sqlite:///", ""))
                with zf.open(m) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                for suffix in ("-wal", "-shm"):
                    Path(str(target) + suffix).unlink(missing_ok=True)
                restored["database"] = True
                continue
            for prefix, base in (("data/", s.data_dir), ("models/", s.models_dir)):
                if name.startswith(prefix) and not m.is_dir():
                    target = (base / name[len(prefix):]).resolve()
                    if base.resolve() not in target.parents:
                        raise ValueError(f"unsafe path {name}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(m) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    restored["files"] += 1
    return restored
