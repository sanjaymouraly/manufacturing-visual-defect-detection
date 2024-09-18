"""Pydantic request schemas (validated inputs) and serializers for ORM objects."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator

from app.models import Inspection, Job, ModelVersion, Product, Review, User

Role = Literal["admin", "inspector", "viewer"]


class LoginIn(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    full_name: str = Field("", max_length=128)
    password: str = Field(min_length=8, max_length=256)
    role: Role = "inspector"


class UserUpdate(BaseModel):
    full_name: str | None = Field(None, max_length=128)
    role: Role | None = None
    is_active: bool | None = None
    password: str | None = Field(None, min_length=8, max_length=256)


class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(min_length=8, max_length=256)


class ReviewIn(BaseModel):
    label: Literal["defective", "normal"]
    defect_type: str = Field("", max_length=64)
    severity: Literal["none", "low", "medium", "high", "critical"] = "none"
    comment: str = Field("", max_length=2000)
    regions: list[list[int]] = Field(default_factory=list, max_length=50)  # [x, y, w, h] in image pixels
    add_to_feedback: bool = True

    @field_validator("regions")
    @classmethod
    def _boxes(cls, v):
        for b in v:
            if len(b) != 4 or any(x < 0 for x in b) or b[2] <= 0 or b[3] <= 0:
                raise ValueError("each region must be [x, y, w, h] with positive width/height")
        return v


class ThresholdUpdate(BaseModel):
    fail_threshold: float = Field(gt=0)
    pass_threshold: float = Field(gt=0)
    seg_threshold: float | None = Field(None, gt=0)
    ood_threshold: float | None = Field(None, gt=0)
    reason: str = Field(min_length=3, max_length=500)


class StatusUpdate(BaseModel):
    status: Literal["candidate", "approved", "archived", "rejected"]
    notes: str = Field("", max_length=1000)


class ActivateIn(BaseModel):
    model_id: int | None
    reason: str = Field("", max_length=500)


class TrainIn(BaseModel):
    dataset_id: int
    algorithm: Literal["patchcore", "padim", "pixelstats"]
    name: str | None = Field(None, max_length=128)
    max_train_images: int | None = Field(None, ge=2, le=5000)
    target_recall: float = Field(0.98, ge=0.5, le=1.0)
    config: dict = Field(default_factory=dict)
    notes: str = Field("", max_length=1000)

    @field_validator("config")
    @classmethod
    def _cfg(cls, v):
        allowed = {"coreset_ratio", "max_memory_bank", "patches_per_image", "n_features", "blur_sigma", "batch_size",
                   "input_size", "work_size"}
        bad = set(v) - allowed
        if bad:
            raise ValueError(f"unsupported config keys: {sorted(bad)}")
        return v


class EvaluateIn(BaseModel):
    dataset_id: int
    model_ids: list[int] = Field(min_length=1, max_length=6)
    split: Literal["val", "test"] = "test"


class RobustnessIn(BaseModel):
    model_id: int
    dataset_id: int | None = None
    degradations: list[str] = Field(default_factory=list)
    severities: list[int] = Field(default_factory=lambda: [1, 3, 5])
    max_samples: int = Field(80, ge=10, le=1000)

    @field_validator("severities")
    @classmethod
    def _sev(cls, v):
        if not v or any(s < 1 or s > 5 for s in v):
            raise ValueError("severities must be within 1..5")
        return sorted(set(v))


class ProductIn(BaseModel):
    key: str = Field(min_length=2, max_length=64, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    name: str = Field(min_length=1, max_length=128)
    description: str = Field("", max_length=1000)


class ProductUpdate(BaseModel):
    name: str | None = Field(None, max_length=128)
    description: str | None = Field(None, max_length=1000)
    policy: dict | None = None
    secondary_model_id: int | None = None
    clear_secondary: bool = False


class ProviderIn(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    kind: Literal["openai", "anthropic", "openrouter", "openai_compatible", "ollama"]
    base_url: str = Field("", max_length=255)
    model: str = Field("", max_length=128)
    api_key: str | None = Field(None, max_length=512)  # write-only
    api_key_env: str = Field("", max_length=64, pattern=r"^[A-Z0-9_]*$")
    timeout_s: float = Field(30.0, ge=1, le=300)
    max_retries: int = Field(2, ge=0, le=6)
    max_tokens: int = Field(600, ge=16, le=8192)
    temperature: float = Field(0.2, ge=0, le=2)
    price_in_per_mtok: float | None = Field(None, ge=0)
    price_out_per_mtok: float | None = Field(None, ge=0)
    enabled: bool = True

    @field_validator("base_url")
    @classmethod
    def _url(cls, v):
        if v:
            HttpUrl(v)
        return v


class CameraIn(BaseModel):
    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    name: str = Field(min_length=1, max_length=128)
    kind: Literal["usb", "ip", "http_snapshot", "folder"]
    source: str = Field(min_length=1, max_length=512)
    product_key: str = ""
    enabled: bool = True


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    product_key: str | None = None
    days: int = Field(30, ge=1, le=3650)
    provider_id: int | None = None


class ReportIn(BaseModel):
    days: int = Field(30, ge=1, le=3650)
    product_key: str | None = None
    batch_code: str | None = None
    max_rows: int = Field(300, ge=1, le=5000)


# ---- serializers ------------------------------------------------------------------------------

def user_out(u: User) -> dict:
    return {"id": u.id, "username": u.username, "full_name": u.full_name, "role": u.role, "is_active": u.is_active,
            "created_at": u.created_at.isoformat() if u.created_at else None,
            "last_login_at": u.last_login_at.isoformat() if u.last_login_at else None,
            "password_change_required": bool(u.password_change_required)}


def _img_urls(i: Inspection) -> dict:
    base = f"/api/inspections/{i.id}/image"
    return {k: (f"{base}/{k}" if getattr(i, f"{k}_path") else None)
            for k in ("display", "heatmap", "overlay", "mask")}


def inspection_out(i: Inspection, detail: bool = False) -> dict:
    d = {
        "id": i.id, "created_at": i.created_at.isoformat(), "product_key": i.product_key, "model_id": i.model_id,
        "model_name": i.model_name, "algorithm": i.algorithm, "source": i.source, "batch_code": i.batch_code,
        "camera_id": i.camera_id, "station": i.station, "external_ref": i.external_ref, "job_id": i.job_id,
        "filename": i.filename, "image_sha256": i.image_sha256, "width": i.width, "height": i.height,
        "score": i.score, "fail_threshold": i.fail_threshold, "pass_threshold": i.pass_threshold,
        "seg_threshold": i.seg_threshold, "normalized_score": i.normalized_score,
        "calibrated_probability": i.calibrated_probability, "ood_score": i.ood_score, "decision": i.decision,
        "severity": i.severity, "reasons": i.reasons or [], "latency_ms": i.latency_ms, "status": i.status,
        "error": i.error, "needs_review": i.needs_review, "review_status": i.review_status,
        "final_label": i.final_label, "duplicate_of": i.duplicate_of, "images_purged": i.images_purged,
        "n_regions": len(i.regions or []), "images": _img_urls(i),
    }
    if detail:
        d.update({"regions": i.regions or [], "quality": i.quality, "defect_type": i.defect_type,
                  "secondary": i.secondary, "reviews": [review_out(r) for r in i.reviews],
                  "reviewed_at": i.reviewed_at.isoformat() if i.reviewed_at else None})
    return d


def review_out(r: Review) -> dict:
    return {"id": r.id, "inspection_id": r.inspection_id, "reviewer_id": r.reviewer_id,
            "reviewer": r.reviewer.username if r.reviewer else None, "label": r.label, "defect_type": r.defect_type,
            "severity": r.severity, "comment": r.comment, "regions": r.regions or [],
            "agrees_with_model": r.agrees_with_model, "created_at": r.created_at.isoformat()}


def model_out(m: ModelVersion, active_for: list[str] | None = None) -> dict:
    t = m.test_metrics or {}
    v = m.validation_metrics or {}
    return {
        "id": m.id, "name": m.name, "product_key": m.product_key, "algorithm": m.algorithm, "version": m.version,
        "status": m.status, "artifact_dir": m.artifact_dir, "weights_sha256": m.weights_sha256,
        "thresholds": m.thresholds, "original_thresholds": m.original_thresholds, "calibrated": bool(m.calibration),
        "calibration": m.calibration, "validation_metrics": v, "test_metrics": t, "training": m.training,
        "dataset": m.dataset, "config": m.config, "history": m.history, "has_classifier": m.has_classifier,
        "classifier_metrics": m.classifier_metrics, "notes": m.notes, "created_at": m.created_at.isoformat(),
        "approved_at": m.approved_at.isoformat() if m.approved_at else None,
        "active_for": active_for or [],
        "summary": {
            "test_auroc": (t.get("ranking") or {}).get("auroc"),
            "test_ap": (t.get("ranking") or {}).get("average_precision"),
            "test_f1": (t.get("at_fail_threshold") or {}).get("f1"),
            "test_miss_rate": (t.get("at_fail_threshold") or {}).get("miss_rate"),
            "test_frr": (t.get("at_fail_threshold") or {}).get("false_rejection_rate"),
            "pixel_auroc": (t.get("pixel") or {}).get("pixel_auroc"),
            "latency_ms": t.get("latency_ms_per_image"),
            "val_auroc": (v.get("ranking") or {}).get("auroc"),
            "n_test": t.get("n"),
        },
    }


def job_out(j: Job) -> dict:
    return {"id": j.id, "type": j.type, "status": j.status, "progress": j.progress, "message": j.message,
            "params": {k: v for k, v in (j.params or {}).items() if k not in ("dir", "zip_path")},
            "result": j.result, "error": j.error, "attempts": j.attempts, "max_attempts": j.max_attempts,
            "cancel_requested": j.cancel_requested, "created_by_id": j.created_by_id,
            "created_at": j.created_at.isoformat(), "started_at": j.started_at.isoformat() if j.started_at else None,
            "finished_at": j.finished_at.isoformat() if j.finished_at else None}


def product_out(p: Product) -> dict:
    return {"key": p.key, "name": p.name, "description": p.description, "active_model_id": p.active_model_id,
            "secondary_model_id": p.secondary_model_id, "policy": p.policy or {}, "synthetic": p.synthetic}
