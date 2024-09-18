"""Core inspection workflow: validate -> deduplicate -> infer -> decide -> persist -> audit."""

from __future__ import annotations

import logging
import threading
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import audit
from app.metrics import METRICS
from app.models import Inspection, ModelVersion, Product, User, new_id, utcnow
from app.services.registry import inspector_cache
from app.services.settings_service import get_setting
from app.services.storage import inspection_dir, safe_filename
from defectml.decision import PolicyConfig, decide
from defectml.imaging import (
    ALLOWED_EXTENSIONS,
    ImageDecodeError,
    QualityLimits,
    decode_image_bytes,
    encode_jpeg,
    resize_max_side,
    sha256_bytes,
)
from defectml.models import ModelLoadError
from defectml.models.backbone import BackboneUnavailable

log = logging.getLogger("defectlens.inspection")
_infer_sem = threading.BoundedSemaphore(1)  # CPU/RAM safeguard: one inference at a time per process


class InspectionInputError(ValueError):
    """Invalid user input (e.g. unknown product, undecodable image) - mapped to HTTP 4xx."""


def effective_policy(db: Session, product: Product) -> PolicyConfig:
    base = get_setting(db, "policy")
    if product.policy:
        base.update({k: v for k, v in product.policy.items() if v is not None})
    return PolicyConfig.from_dict(base)


def _quality_limits(db: Session) -> QualityLimits:
    q = get_setting(db, "quality")
    known = QualityLimits.__dataclass_fields__
    return QualityLimits(**{k: v for k, v in q.items() if k in known})


def find_duplicate(db: Session, sha: str, product_key: str, model_id: int | None) -> Inspection | None:
    hours = int(get_setting(db, "general").get("duplicate_window_hours", 24))
    if hours <= 0:
        return None
    since = utcnow() - timedelta(hours=hours)
    return db.execute(select(Inspection).where(
        Inspection.image_sha256 == sha, Inspection.product_key == product_key, Inspection.model_id == model_id,
        Inspection.status == "ok", Inspection.created_at >= since, Inspection.duplicate_of.is_(None),
    ).order_by(Inspection.created_at.desc()).limit(1)).scalar_one_or_none()


def run_inspection(db: Session, *, data: bytes, filename: str, product_key: str, operator: User | None,
                   model_id: int | None = None, source: str = "upload", batch_code: str = "", camera_id: str = "",
                   station: str = "", external_ref: str = "", job_id: str | None = None,
                   idempotency_key: str | None = None, force: bool = False,
                   persist_decode_errors: bool = False) -> tuple[Inspection, bool]:
    """Inspect one image and persist the result. Returns (inspection, is_duplicate)."""
    if idempotency_key:
        prev = db.execute(select(Inspection).where(Inspection.idempotency_key == idempotency_key)).scalar_one_or_none()
        if prev is not None:
            return prev, True
    product = db.get(Product, product_key)
    if product is None:
        raise InspectionInputError(f"Unknown product '{product_key}'")
    model: ModelVersion | None = None
    mid = model_id or product.active_model_id
    if mid is not None:
        model = db.get(ModelVersion, mid)
        if model is None or model.product_key != product_key:
            raise InspectionInputError("Selected model does not exist for this product")
        if model.status in ("rejected",):
            raise InspectionInputError("Selected model has been rejected and cannot be used")
    sha = sha256_bytes(data)
    if not force:
        dup = find_duplicate(db, sha, product_key, model.id if model else None)
        if dup is not None:
            METRICS.inc("inspections_duplicate_total")
            return dup, True

    insp = Inspection(id=new_id(), created_at=utcnow(), product_key=product_key, model_id=model.id if model else None,
                      model_name=model.name if model else "", algorithm=model.algorithm if model else "",
                      source=source, batch_code=batch_code[:128], camera_id=camera_id[:64], station=station[:64],
                      external_ref=external_ref[:128], job_id=job_id, operator_id=operator.id if operator else None,
                      filename=safe_filename(filename or "image"), image_sha256=sha, idempotency_key=idempotency_key)
    d, rel = inspection_dir(insp.id, insp.created_at)
    ext = Path(insp.filename).suffix.lower()
    ext = ext if ext in ALLOWED_EXTENSIONS else ".bin"

    def fail_safe(reason: str, store_raw: bool = True) -> tuple[Inspection, bool]:
        dec = decide(None, None, error=reason)
        insp.decision, insp.reasons, insp.severity = dec.decision, dec.reasons, dec.severity
        insp.status, insp.error, insp.needs_review, insp.review_status = "error", reason, True, "pending"
        if store_raw:
            (d / f"original{ext}").write_bytes(data)
            insp.image_path = f"{rel}/original{ext}"
        db.add(insp)
        audit.record(db, "inspection.failed_safe", operator, "inspection", insp.id, {"reason": reason})
        db.flush()
        METRICS.inc("inspections_total", decision="REVIEW", status="error")
        return insp, False

    try:
        img = decode_image_bytes(data)
    except ImageDecodeError as exc:
        if not persist_decode_errors:
            import shutil

            shutil.rmtree(d, ignore_errors=True)
            raise InspectionInputError(str(exc)) from exc
        return fail_safe(f"image could not be decoded ({exc})")
    insp.height, insp.width = int(img.shape[0]), int(img.shape[1])
    (d / f"original{ext}").write_bytes(data)
    insp.image_path = f"{rel}/original{ext}"
    disp = resize_max_side(img, 1024)
    (d / "display.jpg").write_bytes(encode_jpeg(disp, 88))
    insp.display_path = f"{rel}/display.jpg"
    if model is None:
        return fail_safe("no active model configured for this product", store_raw=False)
    try:
        inspector = inspector_cache().get(model)
        secondary = None
        if product.secondary_model_id and product.secondary_model_id != model.id:
            sm = db.get(ModelVersion, product.secondary_model_id)
            if sm is not None and sm.status == "approved":
                secondary = inspector_cache().get(sm)
        with _infer_sem:
            out = inspector.inspect(img, policy=effective_policy(db, product), threshold_override=model.thresholds,
                                    secondary=secondary, quality_limits=_quality_limits(db))
    except (ModelLoadError, BackboneUnavailable) as exc:
        log.error("model unavailable: %s", exc)
        return fail_safe(f"model unavailable ({exc})", store_raw=False)
    except MemoryError:
        return fail_safe("out of memory during inference", store_raw=False)
    except Exception as exc:  # model failure must never produce a PASS
        log.exception("inference failed")
        return fail_safe(f"inference error ({exc.__class__.__name__})", store_raw=False)

    if out.heatmap is not None:
        scale_shape = disp.shape[:2]
        hm = cv2.resize(out.heatmap, (scale_shape[1], scale_shape[0]), interpolation=cv2.INTER_LINEAR)
        (d / "heatmap.jpg").write_bytes(encode_jpeg(hm, 85))
        insp.heatmap_path = f"{rel}/heatmap.jpg"
        assert out.overlay is not None
        ov = cv2.resize(out.overlay, (scale_shape[1], scale_shape[0]), interpolation=cv2.INTER_AREA)
        (d / "overlay.jpg").write_bytes(encode_jpeg(ov, 85))
        insp.overlay_path = f"{rel}/overlay.jpg"
    if out.mask is not None:
        m = cv2.resize(out.mask.astype(np.uint8) * 255, (disp.shape[1], disp.shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(d / "mask.png"), m)
        insp.mask_path = f"{rel}/mask.png"
    dec = out.decision
    insp.score = out.score
    insp.fail_threshold = out.thresholds["fail_threshold"]
    insp.pass_threshold = out.thresholds["pass_threshold"]
    insp.seg_threshold = out.thresholds["seg_threshold"]
    insp.normalized_score = dec.normalized_score
    insp.calibrated_probability = out.calibrated_probability
    insp.ood_score = out.ood_score
    insp.decision, insp.severity, insp.reasons = dec.decision, dec.severity, dec.reasons
    insp.regions, insp.quality = out.regions, out.quality
    insp.defect_type, insp.secondary = out.defect_type, out.secondary
    insp.latency_ms = out.latency_ms
    insp.needs_review = dec.needs_review
    insp.review_status = "pending" if dec.needs_review else "none"
    db.add(insp)
    audit.record(db, "inspection.created", operator, "inspection", insp.id,
                 {"decision": insp.decision, "score": insp.score, "model_id": insp.model_id,
                  "fail_threshold": insp.fail_threshold, "pass_threshold": insp.pass_threshold,
                  "image_sha256": sha, "source": source})
    db.flush()
    METRICS.inc("inspections_total", decision=insp.decision, status="ok")
    METRICS.observe("inference_ms", out.latency_ms)
    return insp, False
