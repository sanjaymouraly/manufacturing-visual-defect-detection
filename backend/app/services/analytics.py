"""Analytics computed from persisted inspections and reviews (no synthetic numbers)."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Inspection, ModelVersion, Product, Review, User, utcnow


def _filters(q, product: str | None, since: datetime | None):
    if product:
        q = q.where(Inspection.product_key == product)
    if since:
        q = q.where(Inspection.created_at >= since)
    return q.where(Inspection.duplicate_of.is_(None))


def overview(db: Session, product: str | None = None, days: int = 30) -> dict:
    since = utcnow() - timedelta(days=days)
    q = _filters(select(Inspection.decision, func.count()), product, since).group_by(Inspection.decision)
    counts = {d: n for d, n in db.execute(q).all()}
    total = sum(counts.values())
    err = db.execute(_filters(select(func.count()).select_from(Inspection), product, since)
                     .where(Inspection.status == "error")).scalar() or 0
    pending = db.execute(_filters(select(func.count()).select_from(Inspection), product, None)
                         .where(Inspection.review_status == "pending")).scalar() or 0
    rows = db.execute(_filters(select(Inspection.created_at, Inspection.decision), product, since)).all()
    per_day: dict[str, Counter] = defaultdict(Counter)
    for ts, dec in rows:
        per_day[ts.strftime("%Y-%m-%d")][dec] += 1
    trend = [{"date": d, "PASS": c.get("PASS", 0), "FAIL": c.get("FAIL", 0), "REVIEW": c.get("REVIEW", 0),
              "anomaly_rate": round(c.get("FAIL", 0) / max(sum(c.values()), 1), 4)} for d, c in sorted(per_day.items())]
    recent = db.execute(_filters(select(Inspection), product, None).order_by(Inspection.created_at.desc()).limit(8)).scalars()
    active = []
    prods = db.execute(select(Product).order_by(Product.key)).scalars().all()
    for p in prods:
        m = db.get(ModelVersion, p.active_model_id) if p.active_model_id else None
        active.append({"product": p.key, "product_name": p.name, "synthetic": p.synthetic,
                       "model": ({"id": m.id, "name": m.name, "algorithm": m.algorithm, "version": m.version,
                                  "test_auroc": ((m.test_metrics or {}).get("ranking") or {}).get("auroc")} if m else None)})
    return {
        "window_days": days, "total": total, "pass": counts.get("PASS", 0), "fail": counts.get("FAIL", 0),
        "review": counts.get("REVIEW", 0), "errors": err, "pending_reviews": pending,
        "anomaly_rate": round(counts.get("FAIL", 0) / total, 4) if total else None,
        "review_rate": round(counts.get("REVIEW", 0) / total, 4) if total else None,
        "trend": trend,
        "recent": [{"id": r.id, "created_at": r.created_at.isoformat(), "product_key": r.product_key,
                    "decision": r.decision, "score": r.score, "normalized_score": r.normalized_score,
                    "filename": r.filename, "severity": r.severity} for r in recent],
        "active_models": active,
    }


def full_analytics(db: Session, product: str | None = None, days: int = 30) -> dict:
    since = utcnow() - timedelta(days=days)
    insps = db.execute(_filters(select(Inspection), product, since)).scalars().all()
    # Model performance vs. human ground truth (only reviewed items have ground truth)
    reviewed = [i for i in insps if i.final_label]
    agree: Counter[str] = Counter()
    for i in reviewed:
        model_def = i.decision == "FAIL"
        human_def = i.final_label == "defective"
        if i.decision == "REVIEW":
            agree["review_" + ("defective" if human_def else "normal")] += 1
        elif model_def and human_def:
            agree["true_reject"] += 1
        elif model_def and not human_def:
            agree["false_alarm"] += 1
        elif not model_def and human_def:
            agree["missed_defect"] += 1
        else:
            agree["true_accept"] += 1
    fails = agree["true_reject"] + agree["false_alarm"]
    defect_types: Counter[str] = Counter()
    for i in reviewed:
        if i.final_label == "defective" and i.reviews:
            defect_types[i.reviews[-1].defect_type or "unspecified"] += 1
    # review workload
    rev_rows = db.execute(select(Review, Inspection.created_at, User.username)
                          .join(Inspection, Review.inspection_id == Inspection.id)
                          .join(User, Review.reviewer_id == User.id).where(Review.created_at >= since)).all()
    per_reviewer = Counter(u for _, _, u in rev_rows)
    turnaround = [(r.created_at - c).total_seconds() / 60 for r, c, _ in rev_rows if r.created_at and c]
    pending = sum(1 for i in insps if i.review_status == "pending")
    # throughput & latency
    lat = [i.latency_ms for i in insps if i.latency_ms is not None]
    per_hour: Counter = Counter(i.created_at.strftime("%Y-%m-%d %H:00") for i in insps)
    # batches
    batches: dict[str, Counter] = defaultdict(Counter)
    for i in insps:
        if i.batch_code:
            batches[i.batch_code][i.decision] += 1
    by_model: dict[str, Counter] = defaultdict(Counter)
    for i in insps:
        by_model[i.model_name or "none"][i.decision] += 1
    defect_trend: dict[str, Counter] = defaultdict(Counter)
    for i in insps:
        defect_trend[i.created_at.strftime("%Y-%m-%d")][i.product_key] += int(i.decision == "FAIL")
    return {
        "window_days": days, "n_inspections": len(insps), "n_reviewed": len(reviewed),
        "model_vs_human": dict(agree),
        "false_alarm_rate_among_rejects": round(agree["false_alarm"] / fails, 4) if fails else None,
        "confirmed_defect_types": dict(defect_types),
        "review_workload": {"pending": pending, "completed": len(rev_rows), "per_reviewer": dict(per_reviewer),
                            "median_turnaround_min": round(float(np.median(turnaround)), 1) if turnaround else None},
        "throughput": {"per_hour": [{"hour": h, "count": n} for h, n in sorted(per_hour.items())][-72:],
                       "latency_ms_mean": round(float(np.mean(lat)), 1) if lat else None,
                       "latency_ms_p95": round(float(np.percentile(lat, 95)), 1) if lat else None},
        "batches": [{"batch_code": b, "total": sum(c.values()), "PASS": c.get("PASS", 0), "FAIL": c.get("FAIL", 0),
                     "REVIEW": c.get("REVIEW", 0), "fail_rate": round(c.get("FAIL", 0) / max(sum(c.values()), 1), 4)}
                    for b, c in sorted(batches.items())],
        "by_model": [{"model": k, **dict(v), "total": sum(v.values())} for k, v in sorted(by_model.items())],
        "defect_trend": [{"date": d, **dict(c)} for d, c in sorted(defect_trend.items())],
        "products": sorted({i.product_key for i in insps}),
    }
