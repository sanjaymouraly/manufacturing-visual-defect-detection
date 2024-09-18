"""Model laboratory API: registry, approval, activation/rollback, thresholds, training, evaluation,
robustness testing, ONNX export."""

from __future__ import annotations

import base64
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import audit, jobs
from app.db import get_db
from app.models import DatasetRecord, EvaluationRun, ModelActivation, ModelVersion, Product, User
from app.schemas import (
    ActivateIn,
    EvaluateIn,
    RobustnessIn,
    StatusUpdate,
    ThresholdUpdate,
    TrainIn,
    job_out,
    model_out,
)
from app.security import require_admin, require_any
from app.services import registry
from defectml.evaluation import degradations as D
from defectml.models import ALGORITHM_INFO

router = APIRouter(prefix="/api", tags=["models"])


def _model(db: Session, model_id: int) -> ModelVersion:
    m = db.get(ModelVersion, model_id)
    if m is None:
        raise HTTPException(404, "Model not found")
    return m


def _active_map(db: Session) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for p in db.execute(select(Product)).scalars():
        if p.active_model_id:
            out.setdefault(p.active_model_id, []).append(p.key)
    return out


@router.get("/algorithms")
def algorithms(_: User = Depends(require_any)):
    return [{"key": k, "description": v} for k, v in ALGORITHM_INFO.items()]


@router.get("/models")
def list_models(product_key: str | None = None, _: User = Depends(require_any), db: Session = Depends(get_db)):
    q = select(ModelVersion).order_by(ModelVersion.product_key, ModelVersion.version.desc())
    if product_key:
        q = q.where(ModelVersion.product_key == product_key)
    am = _active_map(db)
    return [model_out(m, am.get(m.id)) for m in db.execute(q).scalars()]


@router.get("/models/{model_id}")
def get_model(model_id: int, _: User = Depends(require_any), db: Session = Depends(get_db)):
    m = _model(db, model_id)
    return model_out(m, _active_map(db).get(m.id))


@router.post("/models/{model_id}/status")
def set_status(model_id: int, body: StatusUpdate, user: User = Depends(require_admin), db: Session = Depends(get_db)):
    m = registry.set_status(db, _model(db, model_id), body.status, user, body.notes)
    db.commit()
    return model_out(m, _active_map(db).get(m.id))


@router.put("/models/{model_id}/thresholds")
def update_thresholds(model_id: int, body: ThresholdUpdate, user: User = Depends(require_admin),
                      db: Session = Depends(get_db)):
    m = _model(db, model_id)
    if body.pass_threshold > body.fail_threshold:
        raise HTTPException(422, "Pass threshold must be <= reject threshold")
    old = dict(m.thresholds)
    new = dict(m.thresholds)
    new.update({k: v for k, v in body.model_dump(exclude={"reason"}).items() if v is not None})
    m.thresholds = new
    audit.record(db, "model.thresholds_changed", user, "model", m.id, {"old": old, "new": new, "reason": body.reason})
    db.commit()
    return model_out(m, _active_map(db).get(m.id))


@router.post("/models/{model_id}/thresholds/reset")
def reset_thresholds(model_id: int, user: User = Depends(require_admin), db: Session = Depends(get_db)):
    m = _model(db, model_id)
    old = dict(m.thresholds)
    m.thresholds = dict(m.original_thresholds)
    audit.record(db, "model.thresholds_reset", user, "model", m.id, {"old": old})
    db.commit()
    return model_out(m, _active_map(db).get(m.id))


@router.post("/products/{product_key}/activate")
def activate(product_key: str, body: ActivateIn, user: User = Depends(require_admin), db: Session = Depends(get_db)):
    p = db.get(Product, product_key)
    if p is None:
        raise HTTPException(404, "Product not found")
    m = _model(db, body.model_id) if body.model_id is not None else None
    registry.activate(db, p, m, user, body.reason)
    db.commit()
    return {"product": product_key, "active_model_id": p.active_model_id}


@router.post("/products/{product_key}/rollback")
def rollback(product_key: str, user: User = Depends(require_admin), db: Session = Depends(get_db)):
    p = db.get(Product, product_key)
    if p is None:
        raise HTTPException(404, "Product not found")
    registry.rollback(db, p, user)
    db.commit()
    return {"product": product_key, "active_model_id": p.active_model_id}


@router.get("/products/{product_key}/activations")
def activations(product_key: str, _: User = Depends(require_any), db: Session = Depends(get_db)):
    rows = db.execute(select(ModelActivation).where(ModelActivation.product_key == product_key)
                      .order_by(ModelActivation.id.desc()).limit(100)).scalars()
    names = dict(db.execute(select(User.id, User.username)).all())
    return [{"id": a.id, "model_id": a.model_id, "previous_model_id": a.previous_model_id, "actor_id": a.actor_id,
             "actor": names.get(a.actor_id) if a.actor_id is not None else None, "reason": a.reason,
             "created_at": a.created_at.isoformat()} for a in rows]


@router.post("/models/train", status_code=202)
def train(body: TrainIn, user: User = Depends(require_admin), db: Session = Depends(get_db)):
    if db.get(DatasetRecord, body.dataset_id) is None:
        raise HTTPException(404, "Dataset not found")
    job = jobs.enqueue(db, "train_model", body.model_dump(), user, max_attempts=1)
    audit.record(db, "model.training_requested", user, "job", job.id, body.model_dump())
    db.commit()
    return job_out(job)


@router.post("/models/evaluate", status_code=202)
def evaluate(body: EvaluateIn, user: User = Depends(require_admin), db: Session = Depends(get_db)):
    rec = db.get(DatasetRecord, body.dataset_id)
    if rec is None:
        raise HTTPException(404, "Dataset not found")
    for mid in body.model_ids:
        m = _model(db, mid)
        if m.product_key != rec.product_key:
            raise HTTPException(422, f"Model {m.id} was trained for '{m.product_key}', dataset is '{rec.product_key}'")
    job = jobs.enqueue(db, "evaluate_models", body.model_dump(), user, max_attempts=1)
    db.commit()
    return job_out(job)


@router.post("/models/{model_id}/classifier", status_code=202)
def train_classifier(model_id: int, user: User = Depends(require_admin), db: Session = Depends(get_db)):
    m = _model(db, model_id)
    if m.algorithm == "pixelstats":
        raise HTTPException(422, "Defect-type classifier requires a feature-based model (PatchCore/PaDiM)")
    job = jobs.enqueue(db, "train_classifier", {"model_id": m.id}, user, max_attempts=1)
    db.commit()
    return job_out(job)


@router.post("/models/{model_id}/export-onnx", status_code=202)
def export_onnx(model_id: int, user: User = Depends(require_admin), db: Session = Depends(get_db)):
    m = _model(db, model_id)
    if m.algorithm not in ("patchcore", "padim"):
        raise HTTPException(422, "ONNX export is available for PatchCore and PaDiM")
    job = jobs.enqueue(db, "export_onnx", {"model_id": m.id}, user, max_attempts=1)
    db.commit()
    return job_out(job)


@router.get("/models/{model_id}/onnx")
def download_onnx(model_id: int, _: User = Depends(require_admin), db: Session = Depends(get_db)):
    p = registry.model_abs_dir(_model(db, model_id)) / "model.onnx"
    if not p.exists():
        raise HTTPException(404, "Not exported yet")
    return FileResponse(p, media_type="application/octet-stream", filename=f"model-{model_id}.onnx")


@router.get("/models/{model_id}/card", response_class=PlainTextResponse)
def model_card(model_id: int, _: User = Depends(require_any), db: Session = Depends(get_db)):
    m = _model(db, model_id)
    t, ds = m.test_metrics or {}, m.dataset or {}
    r, a, px = t.get("ranking") or {}, t.get("at_fail_threshold") or {}, t.get("pixel") or {}

    def f(v):
        return "n/a" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))

    return "\n".join([
        f"# Model card: {m.name} (v{m.version})", "",
        f"- Algorithm: {m.algorithm} — {ALGORITHM_INFO.get(m.algorithm, '')}",
        f"- Product: {m.product_key}; status: {m.status}",
        f"- Training data: {ds.get('name')} ({ds.get('source')}), license {ds.get('license')}"
        + (" — SYNTHETIC data, not indicative of real-world accuracy" if ds.get("synthetic") else ""),
        f"- Weights SHA-256: {m.weights_sha256}",
        f"- Thresholds (validation-selected): {m.original_thresholds}",
        f"- Calibration: {'Platt scaling on validation split' if m.calibration else 'none — scores are not probabilities'}",
        "", "## Held-out test metrics", f"- n = {t.get('n')} ({t.get('n_anomalous')} anomalous)",
        f"- Image AUROC {f(r.get('auroc'))} (95% CI {t.get('auroc_ci95')}), AP {f(r.get('average_precision'))}",
        f"- At reject threshold: recall {f(a.get('recall'))}, precision {f(a.get('precision'))}, F1 {f(a.get('f1'))}, "
        f"miss rate {f(a.get('miss_rate'))}, false-rejection rate {f(a.get('false_rejection_rate'))}",
        f"- Pixel AUROC {f(px.get('pixel_auroc'))}, AUPRO@0.3 {f(px.get('aupro_0.3'))}, Dice {f(px.get('dice'))}",
        f"- Latency {t.get('latency_ms_per_image')} ms/image on {(m.training or {}).get('platform')}",
        "", "## Limitations",
        "- Decision support only; validate on factory-specific data and hardware before automated use.",
        "- Anomaly detectors flag deviations from training normality; unseen benign variation can raise false alarms.",
        "- Backbone: torchvision ResNet18 ImageNet weights (downloaded at runtime; review their terms of use).",
    ])


# ---- evaluations / robustness -------------------------------------------------------------

@router.get("/evaluations")
def list_evaluations(kind: str | None = None, product_key: str | None = None, _: User = Depends(require_any),
                     db: Session = Depends(get_db)):
    q = select(EvaluationRun).order_by(EvaluationRun.id.desc())
    if kind:
        q = q.where(EvaluationRun.kind == kind)
    if product_key:
        q = q.where(EvaluationRun.product_key == product_key)
    out = []
    for e in db.execute(q.limit(200)).scalars():
        res = e.result or {}
        out.append({"id": e.id, "kind": e.kind, "product_key": e.product_key, "dataset_name": e.dataset_name,
                    "model_ids": e.model_ids, "created_at": e.created_at.isoformat(), "job_id": e.job_id,
                    "algorithm": res.get("algorithm"), "model": res.get("model"),
                    "robustness_score": res.get("robustness_score")})
    return out


@router.get("/evaluations/{eval_id}")
def get_evaluation(eval_id: int, _: User = Depends(require_any), db: Session = Depends(get_db)):
    e = db.get(EvaluationRun, eval_id)
    if e is None:
        raise HTTPException(404, "Evaluation not found")
    arts = sorted(p.name for p in Path(e.artifact_dir).glob("*") if p.suffix in (".png", ".jpg")) if e.artifact_dir else []
    return {"id": e.id, "kind": e.kind, "product_key": e.product_key, "dataset_name": e.dataset_name,
            "model_ids": e.model_ids, "created_at": e.created_at.isoformat(), "result": e.result,
            "artifacts": [f"/api/evaluations/{e.id}/artifacts/{a}" for a in arts]}


@router.get("/evaluations/{eval_id}/artifacts/{name}")
def evaluation_artifact(eval_id: int, name: str, _: User = Depends(require_any), db: Session = Depends(get_db)):
    e = db.get(EvaluationRun, eval_id)
    if e is None or not e.artifact_dir:
        raise HTTPException(404, "Not found")
    base = Path(e.artifact_dir).resolve()
    p = (base / Path(name).name).resolve()
    if base not in p.parents or not p.exists() or p.suffix not in (".png", ".jpg"):
        raise HTTPException(404, "Not found")
    return FileResponse(p)


@router.get("/robustness/degradations")
def list_degradations(_: User = Depends(require_any)):
    return {"degradations": list(D.DEGRADATIONS), "groups": D.GROUPS}


@router.post("/robustness", status_code=202)
def run_robustness(body: RobustnessIn, user: User = Depends(require_admin), db: Session = Depends(get_db)):
    _model(db, body.model_id)
    bad = [d for d in body.degradations if d not in D.DEGRADATIONS]
    if bad:
        raise HTTPException(422, f"Unknown degradations: {bad}")
    job = jobs.enqueue(db, "robustness", body.model_dump(), user, max_attempts=1)
    db.commit()
    return job_out(job)


class ShiftIn(BaseModel):
    model_id: int
    max_samples: int = 120


@router.post("/domain-shift", status_code=202)
def run_domain_shift(body: ShiftIn, user: User = Depends(require_admin), db: Session = Depends(get_db)):
    _model(db, body.model_id)
    job = jobs.enqueue(db, "domain_shift", body.model_dump(), user, max_attempts=1)
    db.commit()
    return job_out(job)


@router.get("/robustness/preview")
def robustness_preview(model_id: int, degradation: str, severity: int = 3, sample: int = 0,
                       _: User = Depends(require_admin), db: Session = Depends(get_db)):
    """Apply one degradation to a held-out sample and run the model on clean vs degraded image."""
    from defectml.data.manifest import Dataset
    from defectml.imaging import encode_jpeg, load_image, resize_max_side

    m = _model(db, model_id)
    if degradation not in D.DEGRADATIONS or not 1 <= severity <= 5:
        raise HTTPException(422, "Invalid degradation or severity")
    rec = db.execute(select(DatasetRecord).where(DatasetRecord.name == (m.dataset or {}).get("name"))).scalar_one_or_none()
    if rec is None:
        raise HTTPException(404, "Model's dataset is not registered")
    ds = Dataset(rec.path)
    test = ds.split("test")
    if not test:
        raise HTTPException(404, "Dataset has no test split")
    s = test[sample % len(test)]
    img = load_image(ds.abspath(s.path))
    deg, _ = D.apply(degradation, img, None, severity, seed=sample)
    insp = registry.inspector_cache().get(m)
    clean, bad = insp.model.predict_batch([img, deg])
    th = m.thresholds

    def dec(score):
        return "FAIL" if score >= th["fail_threshold"] else ("PASS" if score < th["pass_threshold"] else "REVIEW")

    def b64(x):
        return "data:image/jpeg;base64," + base64.b64encode(encode_jpeg(resize_max_side(x, 384), 85)).decode()

    return {"sample_path": s.path, "label": s.label, "defect_type": s.defect_type, "n_test": len(test),
            "clean": {"image": b64(img), "score": clean.score, "decision": dec(clean.score)},
            "degraded": {"image": b64(deg), "score": bad.score, "decision": dec(bad.score)},
            "fail_threshold": th["fail_threshold"], "pass_threshold": th["pass_threshold"]}
