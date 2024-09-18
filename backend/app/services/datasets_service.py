"""Dataset registry, custom dataset import and controlled feedback-dataset building."""

from __future__ import annotations

import json
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import audit
from app.config import get_settings
from app.db import SessionLocal
from app.jobs import JobContext, PermanentJobError, handler
from app.models import DatasetRecord, FeedbackSample, Inspection, Review, User, utcnow
from app.services.registry import ensure_product
from app.services.storage import abs_data_path
from defectml.data.folder import DatasetImportError, import_folder
from defectml.data.manifest import LABEL_ANOMALY, LABEL_NORMAL, Dataset, DatasetInfo, Sample, save_dataset
from defectml.data.splits import SplitConfig, make_splits
from defectml.data.validation import validate_dataset
from defectml.imaging import ImageDecodeError, load_image


def kind_for(info: DatasetInfo) -> str:
    if info.synthetic:
        return "synthetic"
    if info.name.startswith("visa-"):
        return "visa"
    if info.source.startswith("feedback"):
        return "feedback"
    return "custom"


def register_dataset(db: Session, path: Path, user: User | None = None, validate: bool = False) -> DatasetRecord:
    ds = Dataset(path)
    rec = db.execute(select(DatasetRecord).where(DatasetRecord.name == ds.info.name)).scalar_one_or_none()
    if rec is None:
        rec = DatasetRecord(name=ds.info.name, product_key=ds.info.product, kind=kind_for(ds.info), path=str(path))
        db.add(rec)
    rec.path = str(Path(path).resolve())
    rec.version = ds.info.version
    rec.sample_count = len(ds.samples)
    rec.manifest_sha256 = ds.info.manifest_sha256
    rec.info = ds.info.to_dict()
    qr = Path(path) / "quality_report.json"
    if validate or not qr.exists():
        rec.quality_report = validate_dataset(ds)
    else:
        rec.quality_report = json.loads(qr.read_text(encoding="utf-8"))
    ensure_product(db, ds.info.product, synthetic=ds.info.synthetic,
                   description="SYNTHETIC demo product" if ds.info.synthetic else ds.info.source)
    db.flush()
    if user is not None:
        audit.record(db, "dataset.registered", user, "dataset", rec.id, {"name": rec.name, "samples": rec.sample_count})
    return rec


def sync_datasets(db: Session) -> list[DatasetRecord]:
    out = []
    for dj in sorted(get_settings().datasets_dir.glob("*/dataset.json")):
        try:
            out.append(register_dataset(db, dj.parent))
        except Exception:  # noqa: S112 - skip unreadable datasets, they are reported via validation
            continue
    return out


@handler("import_dataset")
def import_dataset_job(ctx: JobContext) -> dict:
    p = ctx.params
    zpath = Path(p["zip_path"])
    work = zpath.parent / "extracted"
    work.mkdir(exist_ok=True)
    base = work.resolve()
    with zipfile.ZipFile(zpath) as zf:
        s = get_settings()
        files = [m for m in zf.infolist() if not m.is_dir()]
        if len(files) > s.max_zip_files * 5:
            raise PermanentJobError("Too many files in dataset archive")
        if sum(m.file_size for m in files) > s.max_zip_uncompressed_mb * 1024 * 1024:
            raise PermanentJobError("Archive too large when extracted")
        for m in files:
            target = (work / m.filename).resolve()
            if base not in target.parents:
                raise PermanentJobError(f"Unsafe path in archive: {m.filename}")
            if m.compress_size and m.file_size / m.compress_size > 200:
                raise PermanentJobError("Suspicious compression ratio (possible zip bomb)")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m) as zsrc, open(target, "wb") as dst:
                shutil.copyfileobj(zsrc, dst)
    ctx.progress(0.3, "Archive extracted")
    roots = [d for d in work.iterdir() if d.is_dir() and not d.name.startswith("__MACOSX")]
    src = roots[0] if len(roots) == 1 and not (work / "normal").exists() and not (work / "train").exists() else work
    try:
        out, skipped = import_folder(src, get_settings().datasets_dir, p["name"], p["product_key"],
                                     p.get("license", "user-provided"))
    except DatasetImportError as exc:
        raise PermanentJobError(str(exc)) from exc
    ctx.progress(0.8, "Validating dataset")
    with SessionLocal() as db:
        user = db.get(User, ctx.user_id) if ctx.user_id else None
        rec = register_dataset(db, out, user, validate=True)
        db.commit()
        rid = rec.id
    shutil.rmtree(zpath.parent, ignore_errors=True)
    return {"dataset_id": rid, "name": p["name"], "skipped": skipped[:50]}


def _boxes_to_mask(regions: list, shape: tuple[int, int], src_w: int, src_h: int) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    sx, sy = shape[1] / max(src_w, 1), shape[0] / max(src_h, 1)
    for r in regions:
        bbox = r.get("bbox") if isinstance(r, dict) else r
        if not bbox or len(bbox) != 4:
            continue
        x, y, w, h = bbox
        cv2.rectangle(m, (int(x * sx), int(y * sy)), (int((x + w) * sx), int((y + h) * sy)), 255, -1)
    return m


@handler("build_feedback_dataset")
def build_feedback_dataset_job(ctx: JobContext) -> dict:
    """Create a NEW dataset version = base dataset + approved, reviewed feedback samples.

    Provenance (inspection id, review id, reviewer) is stored per sample in ``source_id``.
    Reviewer-drawn boxes become (coarse) masks and are labelled as such in dataset.json.
    Training on the result produces a *candidate* model that still needs evaluation/approval.
    """
    p = ctx.params
    s = get_settings()
    with SessionLocal() as db:
        base = db.get(DatasetRecord, p["base_dataset_id"])
        if base is None:
            raise PermanentJobError("Base dataset not found")
        fbs = db.execute(select(FeedbackSample).where(FeedbackSample.product_key == base.product_key,
                                                      FeedbackSample.status == "approved")).scalars().all()
        if not fbs:
            raise PermanentJobError("No approved feedback samples for this product")
        base_ds = Dataset(base.path)
        version = 1 + len(db.execute(select(DatasetRecord).where(DatasetRecord.kind == "feedback",
                                                                 DatasetRecord.product_key == base.product_key)).all())
        name = f"{base.name}-fb-v{version}"
        out = s.datasets_dir / name
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(base_ds.root, out, ignore=shutil.ignore_patterns("quality_report.json", "preview.jpg"))
        samples = list(base_ds.samples)
        (out / "images").mkdir(exist_ok=True)
        (out / "masks").mkdir(exist_ok=True)
        added = 0
        for i, fb in enumerate(fbs):
            ctx.check_cancel()
            insp = db.get(Inspection, fb.inspection_id)
            rev = db.get(Review, fb.review_id)
            if insp is None or rev is None or not insp.image_path or insp.images_purged:
                continue
            try:
                img = load_image(abs_data_path(insp.image_path))
            except (ImageDecodeError, ValueError):
                continue
            stem = f"FB_{insp.id[:12]}"
            cv2.imwrite(str(out / "images" / f"{stem}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            mask_rel = ""
            if fb.label == "defective" and rev.regions:
                m = _boxes_to_mask(rev.regions, img.shape[:2], insp.width, insp.height)
                mask_rel = f"masks/{stem}.png"
                cv2.imwrite(str(out / mask_rel), m)
            samples.append(Sample(path=f"images/{stem}.png", label=LABEL_ANOMALY if fb.label == "defective" else LABEL_NORMAL,
                                  defect_type=fb.defect_type or "", mask=mask_rel, split="",
                                  source_id=f"feedback:inspection={insp.id};review={rev.id};reviewer={rev.reviewer_id}"))
            fb.status, fb.dataset_id = "used", None
            added += 1
            ctx.progress(0.1 + 0.7 * (i + 1) / len(fbs), f"Added {added} feedback samples")
        new = [x for x in samples if x.source_id.startswith("feedback:")]
        make_splits(new, SplitConfig(seed=int(p.get("seed", 42)), keep_official_train=False))
        info = DatasetInfo(**{**base_ds.info.to_dict(), "name": name, "version": f"{base_ds.info.version}+fb{version}",
                              "source": f"feedback dataset built from {base.name}",
                              "description": f"{base.name} plus {added} reviewer-verified feedback samples. "
                                             "Feedback masks are coarse reviewer boxes.",
                              "created_at": datetime.now(UTC).isoformat()})
        save_dataset(out, info, samples)
        user = db.get(User, ctx.user_id) if ctx.user_id else None
        rec = register_dataset(db, out, user, validate=True)
        rec.kind = "feedback"
        rec.notes = f"Built from {base.name} + {added} approved feedback samples"
        for fb in fbs:
            if fb.status == "used":
                fb.dataset_id = rec.id
        audit.record(db, "dataset.feedback_version_built", user, "dataset", rec.id,
                     {"base": base.name, "added": added, "feedback_ids": [f.id for f in fbs]})
        db.commit()
        return {"dataset_id": rec.id, "name": name, "added": added}


def decide_feedback(db: Session, fb: FeedbackSample, approve: bool, user: User) -> FeedbackSample:
    if fb.status not in ("pending", "approved", "rejected"):
        raise ValueError("Feedback sample already used in a dataset version")
    fb.status = "approved" if approve else "rejected"
    fb.decided_by_id, fb.decided_at = user.id, utcnow()
    audit.record(db, "feedback.decided", user, "feedback", fb.id, {"status": fb.status})
    return fb
