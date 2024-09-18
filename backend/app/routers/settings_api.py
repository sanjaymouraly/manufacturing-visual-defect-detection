"""Settings: global settings, products & policies, cameras, AI providers, retention, audit, system."""

from __future__ import annotations

import platform

from fastapi import APIRouter, Depends, Form, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import audit, jobs
from app.config import get_settings
from app.db import get_db
from app.models import AuditEvent, Inspection, LLMUsage, ModelVersion, Product, ProviderConfig, User
from app.schemas import CameraIn, ProductIn, ProductUpdate, ProviderIn, inspection_out, job_out, product_out
from app.security import encrypt_secret, mask_secret, require_admin, require_any, require_operator
from app.services import camera as cam_service
from app.services import llm
from app.services.inspection import InspectionInputError, run_inspection
from app.services.registry import inspector_cache
from app.services.settings_service import DEFAULTS, all_settings, get_setting, set_setting
from defectml.decision import PolicyConfig

router = APIRouter(prefix="/api", tags=["settings"])


@router.get("/settings")
def read_settings(_: User = Depends(require_any), db: Session = Depends(get_db)):
    return all_settings(db)


@router.put("/settings/{key}")
def write_settings(key: str, value: dict, user: User = Depends(require_admin), db: Session = Depends(get_db)):
    if key not in DEFAULTS:
        raise HTTPException(404, "Unknown settings section")
    if key == "policy":
        PolicyConfig.from_dict(value)  # validates field names / types
    if (key == "llm" and value.get("active_provider_id") is not None
            and db.get(ProviderConfig, int(value["active_provider_id"])) is None):
        raise HTTPException(422, "Provider not found")
    if key == "camera" and "cameras" in value:
        value["cameras"] = [CameraIn(**c).model_dump() for c in value["cameras"]]
    if key == "retention":
        for k in ("image_days", "inspection_days", "job_days"):
            if k in value and (not isinstance(value[k], int) or value[k] < 0):
                raise HTTPException(422, f"{k} must be a non-negative integer")
    old = get_setting(db, key)
    new = set_setting(db, key, value, user)
    audit.record(db, "settings.updated", user, "settings", key, {"old": old, "new": new})
    db.commit()
    return new


# ---- products ---------------------------------------------------------------------------------

@router.get("/products")
def list_products(_: User = Depends(require_any), db: Session = Depends(get_db)):
    out = []
    for p in db.execute(select(Product).order_by(Product.key)).scalars():
        d = product_out(p)
        m = db.get(ModelVersion, p.active_model_id) if p.active_model_id else None
        d["active_model"] = {"id": m.id, "name": m.name, "algorithm": m.algorithm, "version": m.version,
                             "thresholds": m.thresholds} if m else None
        out.append(d)
    return out


@router.post("/products", status_code=201)
def create_product(body: ProductIn, user: User = Depends(require_admin), db: Session = Depends(get_db)):
    if db.get(Product, body.key):
        raise HTTPException(409, "Product key exists")
    p = Product(key=body.key, name=body.name, description=body.description)
    db.add(p)
    audit.record(db, "product.created", user, "product", p.key, body.model_dump())
    db.commit()
    return product_out(p)


@router.patch("/products/{key}")
def update_product(key: str, body: ProductUpdate, user: User = Depends(require_admin), db: Session = Depends(get_db)):
    p = db.get(Product, key)
    if p is None:
        raise HTTPException(404, "Product not found")
    if body.name is not None:
        p.name = body.name
    if body.description is not None:
        p.description = body.description
    if body.policy is not None:
        PolicyConfig.from_dict({**PolicyConfig().to_dict(), **body.policy})
        p.policy = body.policy
    if body.clear_secondary:
        p.secondary_model_id = None
    elif body.secondary_model_id is not None:
        m = db.get(ModelVersion, body.secondary_model_id)
        if m is None or m.product_key != key or m.status != "approved":
            raise HTTPException(422, "Secondary model must be an approved model of this product")
        p.secondary_model_id = m.id
    audit.record(db, "product.updated", user, "product", key, body.model_dump(exclude_none=True))
    db.commit()
    return product_out(p)


# ---- cameras (server-side adapters) ------------------------------------------------------------

@router.post("/cameras/{camera_id}/inspect", status_code=201)
def camera_inspect(camera_id: str, product_key: str = Form(""), batch_code: str = Form(""),
                   user: User = Depends(require_operator), db: Session = Depends(get_db)):
    cams = {c["id"]: c for c in get_setting(db, "camera")["cameras"]}
    cam = cams.get(camera_id)
    if cam is None or not cam.get("enabled", True):
        raise HTTPException(404, "Camera not configured or disabled")
    try:
        data = cam_service.snapshot(cam)
    except cam_service.CameraError as exc:
        raise HTTPException(503, f"Camera error: {exc}") from exc
    product = product_key or cam.get("product_key") or ""
    try:
        insp, dup = run_inspection(db, data=data, filename=f"{camera_id}.jpg", product_key=product, operator=user,
                                   source="camera", camera_id=camera_id, batch_code=batch_code, force=True,
                                   persist_decode_errors=True)
        db.commit()
    except InspectionInputError as exc:
        db.rollback()
        raise HTTPException(422, str(exc)) from exc
    return {**inspection_out(insp, detail=True), "duplicate": dup}


@router.post("/cameras/{camera_id}/test")
def camera_test(camera_id: str, _: User = Depends(require_admin), db: Session = Depends(get_db)):
    cams = {c["id"]: c for c in get_setting(db, "camera")["cameras"]}
    if camera_id not in cams:
        raise HTTPException(404, "Camera not configured")
    try:
        data = cam_service.snapshot(cams[camera_id])
    except cam_service.CameraError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "bytes": len(data)}


# ---- AI providers --------------------------------------------------------------------------

def provider_out(p: ProviderConfig) -> dict:
    from app.services.llm import api_key_for

    key = api_key_for(p)
    return {"id": p.id, "name": p.name, "kind": p.kind, "base_url": p.base_url,
            "effective_base_url": p.base_url or llm.DEFAULT_BASE_URLS.get(p.kind, ""), "model": p.model,
            "api_key_set": bool(key), "api_key_masked": mask_secret(key), "api_key_env": p.api_key_env,
            "timeout_s": p.timeout_s, "max_retries": p.max_retries, "max_tokens": p.max_tokens,
            "temperature": p.temperature, "price_in_per_mtok": p.price_in_per_mtok,
            "price_out_per_mtok": p.price_out_per_mtok, "enabled": p.enabled, "external": p.kind not in llm.LOCAL_KINDS}


@router.get("/providers")
def list_providers(_: User = Depends(require_admin), db: Session = Depends(get_db)):
    usage = db.execute(select(LLMUsage.provider_name, func.count(), func.sum(LLMUsage.tokens_in),
                              func.sum(LLMUsage.tokens_out), func.sum(LLMUsage.cost_usd))
                       .group_by(LLMUsage.provider_name)).all()
    return {"providers": [provider_out(p) for p in db.execute(select(ProviderConfig).order_by(ProviderConfig.id)).scalars()],
            "kinds": llm.PROVIDER_KINDS, "default_base_urls": llm.DEFAULT_BASE_URLS,
            "active_provider_id": get_setting(db, "llm").get("active_provider_id"),
            "usage": [{"provider": n, "calls": c, "tokens_in": ti or 0, "tokens_out": to or 0,
                       "cost_usd": round(cost, 6) if cost is not None else None} for n, c, ti, to, cost in usage]}


@router.post("/providers", status_code=201)
def create_provider(body: ProviderIn, user: User = Depends(require_admin), db: Session = Depends(get_db)):
    if db.execute(select(ProviderConfig).where(ProviderConfig.name == body.name)).scalar_one_or_none():
        raise HTTPException(409, "Provider name exists")
    if body.kind == "openai_compatible" and not body.base_url:
        raise HTTPException(422, "Custom OpenAI-compatible providers need a base URL")
    data = body.model_dump(exclude={"api_key"})
    p = ProviderConfig(**data, api_key_enc=encrypt_secret(body.api_key or ""))
    db.add(p)
    db.flush()
    audit.record(db, "provider.created", user, "provider", p.id, {"name": p.name, "kind": p.kind, "model": p.model})
    db.commit()
    return provider_out(p)


@router.put("/providers/{provider_id}")
def update_provider(provider_id: int, body: ProviderIn, user: User = Depends(require_admin),
                    db: Session = Depends(get_db)):
    p = db.get(ProviderConfig, provider_id)
    if p is None:
        raise HTTPException(404, "Provider not found")
    for k, v in body.model_dump(exclude={"api_key"}).items():
        setattr(p, k, v)
    if body.api_key is not None:  # empty string clears the stored key; omitted keeps it
        p.api_key_enc = encrypt_secret(body.api_key)
    audit.record(db, "provider.updated", user, "provider", p.id, {"name": p.name, "kind": p.kind, "model": p.model,
                                                                  "key_changed": body.api_key is not None})
    db.commit()
    return provider_out(p)


@router.delete("/providers/{provider_id}")
def delete_provider(provider_id: int, user: User = Depends(require_admin), db: Session = Depends(get_db)):
    p = db.get(ProviderConfig, provider_id)
    if p is None:
        raise HTTPException(404, "Provider not found")
    llm_set = get_setting(db, "llm")
    if llm_set.get("active_provider_id") == provider_id:
        set_setting(db, "llm", {"active_provider_id": None}, user)
    p.enabled = False
    p.api_key_enc = ""
    p.name = f"{p.name} (deleted {p.id})"
    audit.record(db, "provider.deleted", user, "provider", provider_id)
    db.commit()
    return {"ok": True}


@router.post("/providers/{provider_id}/test")
def test_provider(provider_id: int, _: User = Depends(require_admin), db: Session = Depends(get_db)):
    p = db.get(ProviderConfig, provider_id)
    if p is None:
        raise HTTPException(404, "Provider not found")
    import time

    t0 = time.perf_counter()
    try:
        models = llm.list_models(p)
        out: dict = {"ok": True, "models_found": len(models), "latency_ms": round((time.perf_counter() - t0) * 1000)}
        if p.model:
            c = llm.complete(p, "Reply with the single word OK.", "Connection test")
            out.update({"completion_ok": True, "reply": c.text[:40], "tokens_in": c.tokens_in, "tokens_out": c.tokens_out,
                        "cost_usd": llm.estimate_cost(p, c.tokens_in, c.tokens_out)})
        return out
    except llm.LLMError as exc:
        return {"ok": False, "error": str(exc), "latency_ms": round((time.perf_counter() - t0) * 1000)}


@router.get("/providers/{provider_id}/models")
def provider_models(provider_id: int, _: User = Depends(require_admin), db: Session = Depends(get_db)):
    p = db.get(ProviderConfig, provider_id)
    if p is None:
        raise HTTPException(404, "Provider not found")
    try:
        return {"models": llm.list_models(p)}
    except llm.LLMError as exc:
        raise HTTPException(502, str(exc)) from exc


# ---- retention, audit, system -------------------------------------------------------------

@router.post("/retention/run", status_code=202)
def run_retention(dry_run: bool = False, user: User = Depends(require_admin), db: Session = Depends(get_db)):
    job = jobs.enqueue(db, "retention", {"dry_run": dry_run}, user, max_attempts=1)
    db.commit()
    return job_out(job)


@router.get("/audit")
def audit_log(action: str | None = None, entity_type: str | None = None, actor: str | None = None, limit: int = 200,
              _: User = Depends(require_admin), db: Session = Depends(get_db)):
    q = select(AuditEvent).order_by(AuditEvent.id.desc())
    if action:
        q = q.where(AuditEvent.action.like(f"{action}%"))
    if entity_type:
        q = q.where(AuditEvent.entity_type == entity_type)
    if actor:
        q = q.where(AuditEvent.actor_name == actor)
    rows = db.execute(q.limit(min(limit, 2000))).scalars()
    return [{"id": e.id, "created_at": e.created_at.isoformat(), "actor": e.actor_name, "action": e.action,
             "entity_type": e.entity_type, "entity_id": e.entity_id, "details": e.details, "hash": e.hash[:16]}
            for e in rows]


@router.get("/audit/verify")
def audit_verify(_: User = Depends(require_admin), db: Session = Depends(get_db)):
    return audit.verify_chain(db)


@router.get("/system")
def system_info(_: User = Depends(require_any), db: Session = Depends(get_db)):
    s = get_settings()
    try:
        import torch

        torch_info = {"version": torch.__version__, "cuda_available": torch.cuda.is_available(),
                      "threads": torch.get_num_threads()}
    except Exception:
        torch_info = None
    return {"env": s.env, "database": s.database_url.split("://")[0], "worker_mode": s.worker_mode,
            "device_setting": s.device, "torch": torch_info, "python": platform.python_version(),
            "platform": platform.platform(), "loaded_models": inspector_cache().loaded(),
            "limits": {"max_upload_mb": s.max_upload_mb, "max_zip_mb": s.max_zip_mb, "max_zip_files": s.max_zip_files},
            "inspections": db.execute(select(func.count()).select_from(Inspection)).scalar()}
