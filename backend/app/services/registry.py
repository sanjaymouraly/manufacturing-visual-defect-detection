"""Model registry: registration of trained artefacts, approval workflow, activation/rollback,
and a bounded LRU cache of loaded inspectors."""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import audit
from app.config import get_settings
from app.models import ModelActivation, ModelVersion, Product, User, utcnow
from defectml.inference import Inspector

log = logging.getLogger("defectlens.registry")


def _rel(path: Path) -> str:
    base = get_settings().models_dir
    try:
        return Path(path).resolve().relative_to(base).as_posix()
    except ValueError:
        return Path(path).resolve().as_posix()


def model_abs_dir(m: ModelVersion) -> Path:
    p = Path(m.artifact_dir)
    return p if p.is_absolute() else get_settings().models_dir / p


def ensure_product(db: Session, key: str, name: str | None = None, synthetic: bool = False,
                   description: str = "") -> Product:
    p = db.get(Product, key)
    if p is None:
        p = Product(key=key, name=name or key, synthetic=synthetic, description=description)
        db.add(p)
        db.flush()
    return p


def register_model_dir(db: Session, model_dir: Path, status: str = "candidate", user: User | None = None,
                       notes: str = "") -> ModelVersion:
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    rel = _rel(model_dir)
    existing = db.execute(select(ModelVersion).where(ModelVersion.artifact_dir == rel)).scalar_one_or_none()
    if existing is not None:
        return existing
    product = meta["product"]
    ds = meta.get("dataset") or {}
    ensure_product(db, product, synthetic=bool(ds.get("synthetic")),
                   description=("SYNTHETIC demo product" if ds.get("synthetic") else ds.get("source", "")))
    version = (db.execute(select(func.max(ModelVersion.version)).where(ModelVersion.product_key == product)).scalar()
               or 0) + 1
    clf_eval = None
    if (model_dir / "classifier" / "evaluation.json").exists():
        clf_eval = json.loads((model_dir / "classifier" / "evaluation.json").read_text(encoding="utf-8"))
    m = ModelVersion(
        name=meta["name"], product_key=product, algorithm=meta["algorithm"], version=version, artifact_dir=rel,
        weights_sha256=meta.get("weights_sha256", ""), status=status, thresholds=dict(meta["thresholds"]),
        original_thresholds=dict(meta["thresholds"]), calibration=meta.get("calibration"),
        validation_metrics=meta.get("validation_metrics"), training=meta.get("training"), dataset=ds,
        config=meta.get("config"), history=meta.get("history"),
        has_classifier=(model_dir / "classifier" / "classifier.json").exists(), classifier_metrics=clf_eval,
        notes=notes,
    )
    if (model_dir / "test_metrics.json").exists():
        m.test_metrics = json.loads((model_dir / "test_metrics.json").read_text(encoding="utf-8"))
    db.add(m)
    db.flush()
    audit.record(db, "model.registered", user, "model", m.id, {"name": m.name, "artifact": rel, "version": version})
    return m


def sync_models(db: Session) -> list[ModelVersion]:
    """Register any model directories present on disk but missing from the database."""
    out = []
    for mj in sorted(get_settings().models_dir.glob("*/model.json")):
        try:
            out.append(register_model_dir(db, mj.parent))
        except Exception as exc:
            log.warning("could not register %s: %s", mj.parent, exc)
    return out


def set_status(db: Session, m: ModelVersion, status: str, user: User, notes: str = "") -> ModelVersion:
    if status not in ("candidate", "approved", "archived", "rejected"):
        raise HTTPException(422, "Invalid status")
    active_for = db.execute(select(Product).where(Product.active_model_id == m.id)).scalars().first()
    if active_for is not None and status in ("archived", "rejected"):
        raise HTTPException(409, "Model is active; activate another model before archiving/rejecting it")
    prev = m.status
    m.status = status
    if status == "approved":
        m.approved_by_id, m.approved_at = user.id, utcnow()
    if notes:
        m.notes = (m.notes + "\n" if m.notes else "") + f"[{utcnow():%Y-%m-%d}] {user.username}: {notes}"
    audit.record(db, "model.status_changed", user, "model", m.id, {"from": prev, "to": status, "notes": notes})
    return m


def activate(db: Session, product: Product, m: ModelVersion | None, user: User, reason: str = "") -> Product:
    if m is not None:
        if m.product_key != product.key:
            raise HTTPException(422, "Model belongs to a different product")
        if m.status != "approved":
            raise HTTPException(409, "Only approved models can be activated (approve it in the Model Lab first)")
    prev = product.active_model_id
    product.active_model_id = m.id if m else None
    db.add(ModelActivation(product_key=product.key, model_id=m.id if m else None, previous_model_id=prev,
                           actor_id=user.id, reason=reason))
    audit.record(db, "model.activated", user, "product", product.key,
                 {"model_id": m.id if m else None, "previous_model_id": prev, "reason": reason})
    return product


def rollback(db: Session, product: Product, user: User) -> Product:
    last = db.execute(select(ModelActivation).where(ModelActivation.product_key == product.key)
                      .order_by(ModelActivation.id.desc()).limit(1)).scalar_one_or_none()
    if last is None or last.previous_model_id is None:
        raise HTTPException(409, "No previous model to roll back to")
    prev = db.get(ModelVersion, last.previous_model_id)
    if prev is None or prev.status not in ("approved",):
        raise HTTPException(409, "Previous model is no longer approved")
    return activate(db, product, prev, user, reason=f"rollback from model {product.active_model_id}")


class InspectorCache:
    def __init__(self, size: int):
        self.size = max(1, size)
        self._d: OrderedDict[tuple, Inspector] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, m: ModelVersion) -> Inspector:
        key = (m.id, m.weights_sha256)
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                return self._d[key]
        insp = Inspector(model_abs_dir(m), device=get_settings().device)  # may raise ModelLoadError
        with self._lock:
            self._d[key] = insp
            while len(self._d) > self.size:
                self._d.popitem(last=False)
        return insp

    def clear(self) -> None:
        with self._lock:
            self._d.clear()

    def loaded(self) -> list[int]:
        with self._lock:
            return [k[0] for k in self._d]


_cache: InspectorCache | None = None


def inspector_cache() -> InspectorCache:
    global _cache
    if _cache is None:
        _cache = InspectorCache(get_settings().model_cache_size)
    return _cache
