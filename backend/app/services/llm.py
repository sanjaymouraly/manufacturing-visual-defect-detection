"""Optional LLM integrations (summaries / explanations / Q&A over inspection statistics).

The computer-vision models remain the only source of inspection decisions. LLM output is
always labelled as model-generated, receives only structured facts (never images), and is
instructed not to invent observations. ``local`` mode uses deterministic templates and needs no
API key, network access or payment.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from dataclasses import dataclass

import httpx
from sqlalchemy.orm import Session

from app.models import Inspection, LLMUsage, ProviderConfig
from app.security import decrypt_secret
from app.services.analytics import full_analytics, overview
from app.services.settings_service import get_setting

DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "ollama": "http://localhost:11434/v1",
    "openai_compatible": "",
}
PROVIDER_KINDS = list(DEFAULT_BASE_URLS)
LOCAL_KINDS = {"ollama"}  # runs on user's own hardware - no external data transfer by default

SYSTEM_PROMPT = (
    "You are a manufacturing quality-control reporting assistant. Use ONLY the facts in the provided JSON. "
    "Never invent measurements, defect types, labels, counts or confidence values. Never change, override or "
    "second-guess the computer-vision decision; human inspectors make final decisions. Raw anomaly scores are "
    "unitless and are NOT probabilities unless a field explicitly says 'calibrated_probability'. If information "
    "is missing, say it is not available. Be concise (<= 180 words), use plain English and short bullet points."
)


class LLMError(RuntimeError):
    pass


class PrivacyBlocked(LLMError):
    pass


@dataclass
class Completion:
    text: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    model: str


def api_key_for(cfg: ProviderConfig) -> str:
    if cfg.api_key_env:
        return os.environ.get(cfg.api_key_env, "")
    return decrypt_secret(cfg.api_key_enc)


def base_url_for(cfg: ProviderConfig) -> str:
    url = (cfg.base_url or DEFAULT_BASE_URLS.get(cfg.kind, "")).rstrip("/")
    if not url:
        raise LLMError("Base URL is required for this provider")
    if not url.startswith(("http://", "https://")):
        raise LLMError("Base URL must start with http:// or https://")
    return url


def _headers(cfg: ProviderConfig) -> dict:
    key = api_key_for(cfg)
    if cfg.kind == "anthropic":
        return {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    h = {"content-type": "application/json"}
    if key:
        h["authorization"] = f"Bearer {key}"
    if cfg.kind == "openrouter":
        h["x-title"] = "DefectLens"
    return h


def _request(cfg: ProviderConfig, method: str, path: str, json_body: dict | None = None,
             transport: httpx.BaseTransport | None = None) -> dict:
    url = base_url_for(cfg) + path
    last: Exception | None = None
    for attempt in range(max(0, cfg.max_retries) + 1):
        try:
            with httpx.Client(timeout=cfg.timeout_s, transport=transport) as c:
                r = c.request(method, url, headers=_headers(cfg), json=json_body)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < cfg.max_retries:
                time.sleep(min(4.0, 0.5 * 2 ** attempt))
                continue
            if r.status_code >= 400:
                msg = r.text[:300]
                with contextlib.suppress(Exception):
                    msg = json.dumps(r.json().get("error", r.json()))[:300]
                raise LLMError(f"HTTP {r.status_code}: {msg}")
            return r.json()
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last = exc
            if attempt < cfg.max_retries:
                time.sleep(min(4.0, 0.5 * 2 ** attempt))
                continue
    raise LLMError(f"Connection failed: {last.__class__.__name__ if last else 'unknown'}")


def list_models(cfg: ProviderConfig, transport: httpx.BaseTransport | None = None) -> list[dict]:
    if cfg.kind == "ollama":
        try:
            root = base_url_for(cfg).removesuffix("/v1")
            with httpx.Client(timeout=cfg.timeout_s, transport=transport) as c:
                r = c.get(root + "/api/tags")
            if r.status_code == 200:
                return [{"id": m["name"]} for m in r.json().get("models", [])]
        except httpx.HTTPError:
            pass
    data = _request(cfg, "GET", "/models", transport=transport)
    out = []
    for m in data.get("data", []):
        item = {"id": m.get("id")}
        pr = m.get("pricing") or {}
        try:  # OpenRouter returns USD per token
            if pr.get("prompt") is not None:
                item["price_in_per_mtok"] = round(float(pr["prompt"]) * 1e6, 4)
                item["price_out_per_mtok"] = round(float(pr["completion"]) * 1e6, 4)
        except (TypeError, ValueError):
            pass
        if m.get("context_length"):
            item["context_length"] = m["context_length"]
        out.append(item)
    return sorted(out, key=lambda x: x["id"] or "")


def complete(cfg: ProviderConfig, system: str, user: str, transport: httpx.BaseTransport | None = None) -> Completion:
    if not cfg.model:
        raise LLMError("No model selected for this provider")
    t0 = time.perf_counter()
    if cfg.kind == "anthropic":
        body = {"model": cfg.model, "max_tokens": cfg.max_tokens, "temperature": cfg.temperature, "system": system,
                "messages": [{"role": "user", "content": user}]}
        d = _request(cfg, "POST", "/messages", body, transport)
        text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
        u = d.get("usage", {})
        tin, tout = int(u.get("input_tokens", 0)), int(u.get("output_tokens", 0))
    else:
        body = {"model": cfg.model, "max_tokens": cfg.max_tokens, "temperature": cfg.temperature,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        d = _request(cfg, "POST", "/chat/completions", body, transport)
        try:
            text = d["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Unexpected response format from provider") from exc
        u = d.get("usage") or {}
        tin, tout = int(u.get("prompt_tokens", 0)), int(u.get("completion_tokens", 0))
    return Completion(text.strip(), tin, tout, (time.perf_counter() - t0) * 1000, cfg.model)


def estimate_cost(cfg: ProviderConfig | None, tin: int, tout: int) -> float | None:
    if cfg is None or cfg.price_in_per_mtok is None or cfg.price_out_per_mtok is None:
        return None
    return round(tin / 1e6 * cfg.price_in_per_mtok + tout / 1e6 * cfg.price_out_per_mtok, 6)


# ---- facts & templates ------------------------------------------------------------------------

def inspection_facts(i: Inspection, include_filename: bool = False) -> dict:
    f = {
        "inspection_id": i.id, "timestamp_utc": i.created_at.isoformat(), "product": i.product_key,
        "decision": i.decision, "severity": i.severity, "raw_anomaly_score": i.score,
        "normalized_score_(score/reject_threshold)": i.normalized_score, "reject_threshold": i.fail_threshold,
        "pass_threshold": i.pass_threshold, "calibrated_probability": i.calibrated_probability,
        "model": i.model_name, "algorithm": i.algorithm, "decision_reasons": i.reasons,
        "image_quality_issues": (i.quality or {}).get("issues", []),
        "localized_regions": [{"bbox_px": r["bbox"], "area_percent": round(100 * r["area_fraction"], 2)}
                              for r in (i.regions or [])[:5]],
        "defect_type_prediction": i.defect_type, "review_status": i.review_status, "final_label": i.final_label,
        "reviews": [{"label": r.label, "defect_type": r.defect_type, "severity": r.severity, "comment": r.comment}
                    for r in i.reviews],
        "status": i.status, "error": i.error or None,
    }
    if include_filename:
        f["filename"] = i.filename
    return f


def template_inspection_summary(f: dict) -> str:
    lines = [f"Inspection {f['inspection_id'][:8]} ({f['product']}): decision {f['decision']}"
             + (f", severity {f['severity']}." if f["severity"] not in (None, "none") else ".")]
    if f["raw_anomaly_score"] is not None:
        lines.append(f"- Raw anomaly score {f['raw_anomaly_score']:.4g} vs. reject threshold {f['reject_threshold']:.4g} "
                     f"(normalised {f['normalized_score_(score/reject_threshold)']:.2f}; 1.00 = threshold).")
    if f["calibrated_probability"] is not None:
        lines.append(f"- Calibrated defect probability (validation-fitted): {f['calibrated_probability']:.2f}.")
    else:
        lines.append("- No calibrated probability is available; the score is a relative measure only.")
    if f["localized_regions"]:
        areas = ", ".join(f"{r['area_percent']}%" for r in f["localized_regions"])
        lines.append(f"- {len(f['localized_regions'])} anomalous region(s) localised (area of image: {areas}).")
    for r in f["decision_reasons"] or []:
        lines.append(f"- Reason: {r}")
    if f["image_quality_issues"]:
        lines.append("- Image quality issues: " + "; ".join(f["image_quality_issues"]))
    if f["final_label"]:
        lines.append(f"- Human review outcome: {f['final_label']}.")
    elif f["review_status"] == "pending":
        lines.append("- Awaiting human review.")
    return "\n".join(lines)


def template_period_summary(ov: dict, an: dict) -> str:
    if not ov["total"]:
        return "No inspections were recorded in the selected period."
    s = [f"{ov['total']} units inspected in the last {ov['window_days']} days: {ov['pass']} PASS, {ov['fail']} FAIL, "
         f"{ov['review']} REVIEW ({ov['errors']} could not be inspected)."]
    if ov["anomaly_rate"] is not None:
        s.append(f"- Rejection (FAIL) rate {100 * ov['anomaly_rate']:.1f}%; review rate {100 * (ov['review_rate'] or 0):.1f}%.")
    mv = an["model_vs_human"]
    if mv:
        parts = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in sorted(mv.items()))
        s.append(f"- Of {an['n_reviewed']} reviewed items: {parts}.")
    if an["confirmed_defect_types"]:
        top = sorted(an["confirmed_defect_types"].items(), key=lambda t: -t[1])[:3]
        s.append("- Most frequent reviewer-confirmed defect types: " + ", ".join(f"{k} ({v})" for k, v in top) + ".")
    worst = sorted([b for b in an["batches"] if b["total"] >= 3], key=lambda b: -b["fail_rate"])[:3]
    if worst:
        parts = ", ".join(f"{b['batch_code']} ({100 * b['fail_rate']:.0f}% of {b['total']})" for b in worst)
        s.append(f"- Batches with highest fail rate: {parts}.")
    s.append(f"- {ov['pending_reviews']} item(s) awaiting human review.")
    return "\n".join(s)


def _active_provider(db: Session, provider_id: int | None) -> ProviderConfig | None:
    pid = provider_id if provider_id is not None else get_setting(db, "llm").get("active_provider_id")
    if not pid:
        return None
    cfg = db.get(ProviderConfig, int(pid))
    if cfg is None or not cfg.enabled:
        raise LLMError("Selected AI provider is missing or disabled")
    return cfg


def _check_privacy(db: Session, cfg: ProviderConfig) -> None:
    priv = get_setting(db, "privacy")
    if cfg.kind not in LOCAL_KINDS and not priv.get("allow_external_llm"):
        raise PrivacyBlocked("External AI providers are disabled in Settings > Privacy. Enable 'Allow external AI "
                             "providers' to send inspection metadata (never images) to this provider.")


def _run(db: Session, cfg: ProviderConfig | None, purpose: str, facts: dict, instruction: str, fallback: str) -> dict:
    if cfg is None:
        db.add(LLMUsage(provider_name="local", model="template", purpose=purpose))
        db.commit()
        return {"text": fallback, "generated_by": "deterministic template (local mode, no AI model)",
                "model_generated": False, "provider": "local", "model": None, "cost_usd": 0.0}
    _check_privacy(db, cfg)
    user = f"{instruction}\n\nFACTS (JSON):\n{json.dumps(facts, default=str)[:12000]}"
    try:
        c = complete(cfg, SYSTEM_PROMPT, user)
    except LLMError:
        db.add(LLMUsage(provider_id=cfg.id, provider_name=cfg.name, model=cfg.model, purpose=purpose, success=False))
        db.commit()
        raise
    cost = estimate_cost(cfg, c.tokens_in, c.tokens_out)
    db.add(LLMUsage(provider_id=cfg.id, provider_name=cfg.name, model=c.model, purpose=purpose, tokens_in=c.tokens_in,
                    tokens_out=c.tokens_out, cost_usd=cost, latency_ms=c.latency_ms))
    db.commit()
    return {"text": c.text, "generated_by": f"{cfg.name} / {c.model}", "model_generated": True, "provider": cfg.name,
            "model": c.model, "tokens_in": c.tokens_in, "tokens_out": c.tokens_out, "cost_usd": cost,
            "notice": "AI-generated text based only on recorded inspection data. It does not change any decision.",
            "facts_fallback": fallback}


def summarize_inspection(db: Session, i: Inspection, provider_id: int | None = None) -> dict:
    priv = get_setting(db, "privacy")
    facts = inspection_facts(i, include_filename=bool(priv.get("include_filenames_in_llm_context")))
    cfg = _active_provider(db, provider_id)
    return _run(db, cfg, "inspection_summary", facts,
                "Write a plain-English summary of this inspection result for a shift supervisor.",
                template_inspection_summary(facts))


def summarize_period(db: Session, product: str | None, days: int, provider_id: int | None = None) -> dict:
    ov, an = overview(db, product, days), full_analytics(db, product, days)
    facts = {"overview": {k: v for k, v in ov.items() if k not in ("recent",)}, "analytics": an}
    cfg = _active_provider(db, provider_id)
    return _run(db, cfg, "period_summary", facts,
                "Summarise quality performance and repeated failure patterns for this period.",
                template_period_summary(ov, an))


def answer_question(db: Session, question: str, product: str | None, days: int, provider_id: int | None = None) -> dict:
    ov, an = overview(db, product, days), full_analytics(db, product, days)
    facts = {"overview": {k: v for k, v in ov.items() if k not in ("recent",)}, "analytics": an}
    cfg = _active_provider(db, provider_id)
    fallback = ("Local mode cannot interpret free-form questions (no AI model configured). Relevant recorded "
                "statistics:\n" + template_period_summary(ov, an))
    return _run(db, cfg, "question", facts, f"Answer this question using only the facts: {question[:500]}", fallback)
