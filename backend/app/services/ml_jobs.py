"""Background jobs for model training, evaluation, robustness testing and export."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from app import audit
from app.config import get_settings
from app.db import SessionLocal
from app.jobs import JobContext, PermanentJobError, handler
from app.models import DatasetRecord, EvaluationRun, ModelVersion, User
from app.services.registry import inspector_cache, model_abs_dir, register_model_dir
from defectml.data.manifest import Dataset
from defectml.models import ALGORITHMS


def _dataset(db, dataset_id: int) -> tuple[DatasetRecord, Dataset]:
    rec = db.get(DatasetRecord, dataset_id)
    if rec is None:
        raise PermanentJobError("Dataset not found")
    return rec, Dataset(rec.path)


def _dataset_for_model(db, m: ModelVersion, dataset_id: int | None) -> tuple[DatasetRecord, Dataset]:
    if dataset_id:
        return _dataset(db, dataset_id)
    from sqlalchemy import select

    name = (m.dataset or {}).get("name")
    rec = db.execute(select(DatasetRecord).where(DatasetRecord.name == name)).scalar_one_or_none()
    if rec is None:
        raise PermanentJobError(f"Training dataset '{name}' is not registered")
    return rec, Dataset(rec.path)


@handler("train_model")
def train_job(ctx: JobContext) -> dict:
    from defectml.evaluation.benchmark import evaluate_model
    from defectml.training import train_on_dataset

    p = ctx.params
    algo = p["algorithm"]
    if algo not in ALGORITHMS:
        raise PermanentJobError(f"Unknown algorithm {algo}")
    with SessionLocal() as db:
        rec, ds = _dataset(db, p["dataset_id"])
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^a-z0-9-]+", "-", f"{rec.product_key}-{algo}-{stamp}".lower())
    out = get_settings().models_dir / slug
    cfg = p.get("config") or {}
    cfg.setdefault("device", get_settings().device)
    meta = train_on_dataset(ds, algo, out, name=p.get("name") or slug, config=cfg,
                            max_train_images=p.get("max_train_images"), target_recall=float(p.get("target_recall", 0.98)),
                            progress=lambda f, m: ctx.progress(0.8 * f, m), cancel=ctx.cancelled)
    ctx.check_cancel()
    ctx.progress(0.82, "Evaluating on held-out test split (reporting only)")
    test = evaluate_model(out, ds, "test", progress=lambda f, m: ctx.progress(0.82 + 0.16 * f, "Test: " + m),
                          cancel=ctx.cancelled)
    test.pop("scores", None)
    (out / "test_metrics.json").write_text(json.dumps(test, indent=2, default=float), encoding="utf-8")
    with SessionLocal() as db:
        user = db.get(User, ctx.user_id) if ctx.user_id else None
        m = register_model_dir(db, out, status="candidate", user=user, notes=p.get("notes", ""))
        audit.record(db, "model.trained", user, "model", m.id, {"dataset": rec.name, "algorithm": algo,
                                                                "job_id": ctx.job_id})
        db.commit()
        return {"model_id": m.id, "name": m.name, "val_auroc": meta["validation_metrics"]["ranking"]["auroc"],
                "test_auroc": test["ranking"]["auroc"]}


@handler("evaluate_models")
def evaluate_job(ctx: JobContext) -> dict:
    from defectml.evaluation.benchmark import compare, evaluate_model

    p = ctx.params
    with SessionLocal() as db:
        rec, ds = _dataset(db, p["dataset_id"])
        models = [db.get(ModelVersion, i) for i in p["model_ids"]]
        if any(m is None for m in models):
            raise PermanentJobError("Model not found")
        dirs = [(m.id, model_abs_dir(m)) for m in models if m is not None]
    results = []
    for k, (_mid, d) in enumerate(dirs):
        ctx.check_cancel()
        r = evaluate_model(d, ds, p.get("split", "test"),
                           progress=lambda f, msg, k=k: ctx.progress((k + f) / len(dirs), f"Model {k + 1}/{len(dirs)}: {msg}"),
                           cancel=ctx.cancelled)
        results.append(r)
    art = get_settings().reports_dir / f"benchmark-{ctx.job_id}"
    summary = compare(results, art)
    with SessionLocal() as db:
        run = EvaluationRun(kind="benchmark", product_key=rec.product_key, dataset_name=rec.name,
                            model_ids=[m for m, _ in dirs], job_id=ctx.job_id, artifact_dir=str(art),
                            result={"rows": summary["rows"],
                                    "results": [{k: v for k, v in r.items() if k != "scores"} for r in results]},
                            created_by_id=ctx.user_id)
        db.add(run)
        for (mid, _), r in zip(dirs, results, strict=True):
            m = db.get(ModelVersion, mid)
            if m is not None and (m.dataset or {}).get("name") == rec.name and p.get("split", "test") == "test":
                m.test_metrics = {k: v for k, v in r.items() if k != "scores"}
        db.commit()
        return {"evaluation_id": run.id, "rows": summary["rows"]}


@handler("robustness")
def robustness_job(ctx: JobContext) -> dict:
    from defectml.evaluation.robustness import run_robustness

    p = ctx.params
    with SessionLocal() as db:
        m = db.get(ModelVersion, p["model_id"])
        if m is None:
            raise PermanentJobError("Model not found")
        rec, ds = _dataset_for_model(db, m, p.get("dataset_id"))
        mdir, mid = model_abs_dir(m), m.id
    art = get_settings().reports_dir / f"robustness-{ctx.job_id}"
    r = run_robustness(mdir, ds, p.get("degradations") or None, p.get("severities") or None,
                       max_samples=int(p.get("max_samples", 80)), out_dir=art,
                       progress=ctx.progress, cancel=ctx.cancelled)
    with SessionLocal() as db:
        run = EvaluationRun(kind="robustness", product_key=rec.product_key, dataset_name=rec.name, model_ids=[mid],
                            job_id=ctx.job_id, artifact_dir=str(art), result=r, created_by_id=ctx.user_id)
        db.add(run)
        db.commit()
        return {"evaluation_id": run.id, "robustness_score": r["robustness_score"]}


@handler("domain_shift")
def domain_shift_job(ctx: JobContext) -> dict:
    from defectml.evaluation.robustness import domain_shift_experiment

    p = ctx.params
    with SessionLocal() as db:
        m = db.get(ModelVersion, p["model_id"])
        if m is None:
            raise PermanentJobError("Model not found")
        rec, ds = _dataset_for_model(db, m, p.get("dataset_id"))
        mdir, mid = model_abs_dir(m), m.id
    ctx.progress(0.1, "Running clean / shifted / recalibrated evaluation")
    art = get_settings().reports_dir / f"domainshift-{ctx.job_id}"
    r = domain_shift_experiment(mdir, ds, art, max_samples=int(p.get("max_samples", 120)))
    with SessionLocal() as db:
        run = EvaluationRun(kind="domain_shift", product_key=rec.product_key, dataset_name=rec.name, model_ids=[mid],
                            job_id=ctx.job_id, artifact_dir=str(art), result=r, created_by_id=ctx.user_id)
        db.add(run)
        db.commit()
        return {"evaluation_id": run.id}


@handler("train_classifier")
def classifier_job(ctx: JobContext) -> dict:
    from defectml.classifier_training import train_defect_classifier

    p = ctx.params
    with SessionLocal() as db:
        m = db.get(ModelVersion, p["model_id"])
        if m is None:
            raise PermanentJobError("Model not found")
        _rec, ds = _dataset_for_model(db, m, None)
        mdir = model_abs_dir(m)
    ctx.progress(0.2, "Training defect-type classifier")
    try:
        r = train_defect_classifier(ds, mdir, min_per_class=int(p.get("min_per_class", 5)))
    except ValueError as exc:
        raise PermanentJobError(str(exc)) from exc
    with SessionLocal() as db:
        m = db.get(ModelVersion, p["model_id"])
        if m is None:
            raise PermanentJobError("Model was deleted while the classifier was training")
        m.has_classifier, m.classifier_metrics = True, r
        user = db.get(User, ctx.user_id) if ctx.user_id else None
        audit.record(db, "model.classifier_trained", user, "model", m.id, {"classes": r.get("classes")})
        db.commit()
    inspector_cache().clear()
    return r


@handler("export_onnx")
def export_job(ctx: JobContext) -> dict:
    from defectml.export import export_onnx
    from defectml.imaging import load_image

    p = ctx.params
    with SessionLocal() as db:
        m = db.get(ModelVersion, p["model_id"])
        if m is None:
            raise PermanentJobError("Model not found")
        mdir = model_abs_dir(m)
        try:
            _rec, ds = _dataset_for_model(db, m, None)
            imgs = [load_image(ds.abspath(s.path)) for s in ds.split("test")[:4]]
        except PermanentJobError:
            imgs = None
    ctx.progress(0.3, "Exporting ONNX graph")
    try:
        r = export_onnx(mdir, verify_images=imgs)
    except ValueError as exc:
        raise PermanentJobError(str(exc)) from exc
    r.pop("torch_scores", None)
    r.pop("onnx_scores", None)
    r["onnx_path"] = Path(r["onnx_path"]).name
    return r
