"""Human review queue, review submission (append-only) and the controlled feedback pipeline."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import audit
from app.db import get_db
from app.models import FeedbackSample, Inspection, Review, User, utcnow
from app.schemas import ReviewIn, inspection_out, review_out
from app.security import require_admin, require_any, require_operator
from app.services.datasets_service import decide_feedback

router = APIRouter(prefix="/api", tags=["reviews"])
SEV_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3, "none": 4}


def reason_categories(i: Inspection) -> list[str]:
    cats = []
    text = " ".join(i.reasons or []).lower()
    if i.status == "error" or "could not be performed" in text:
        cats.append("inspection_failed")
    if "borderline" in text:
        cats.append("borderline")
    if "quality" in text:
        cats.append("image_quality")
    if "out-of-distribution" in text:
        cats.append("out_of_distribution")
    if "disagrees" in text:
        cats.append("model_disagreement")
    if "requires inspector confirmation" in text:
        cats.append("high_risk")
    return cats or ["other"]


@router.get("/reviews/queue")
def queue(product_key: str | None = None, category: str | None = None, limit: int = 200,
          _: User = Depends(require_any), db: Session = Depends(get_db)):
    q = select(Inspection).where(Inspection.review_status == "pending")
    if product_key:
        q = q.where(Inspection.product_key == product_key)
    rows = db.execute(q.order_by(Inspection.created_at.asc()).limit(2000)).scalars().all()
    items = []
    for i in rows:
        cats = reason_categories(i)
        if category and category not in cats:
            continue
        items.append({**inspection_out(i), "categories": cats})
    items.sort(key=lambda x: (0 if "inspection_failed" in x["categories"] else 1, SEV_RANK.get(x["severity"], 5),
                              x["created_at"]))
    counts: dict[str, int] = {}
    for it in items:
        for c in it["categories"]:
            counts[c] = counts.get(c, 0) + 1
    return {"items": items[:limit], "total": len(items), "categories": counts}


@router.post("/reviews/{inspection_id}", status_code=201)
def submit_review(inspection_id: str, body: ReviewIn, user: User = Depends(require_operator),
                  db: Session = Depends(get_db)):
    i = db.get(Inspection, inspection_id)
    if i is None:
        raise HTTPException(404, "Inspection not found")
    if body.label == "defective" and body.severity == "none":
        raise HTTPException(422, "Assign a severity for defective items")
    if i.width and i.height:
        for x, y, w, h in body.regions:
            if x + w > i.width + 1 or y + h > i.height + 1:
                raise HTTPException(422, "Region outside image bounds")
    model_says_defect = i.decision == "FAIL"
    agrees = (i.decision == "REVIEW") or (model_says_defect == (body.label == "defective"))
    prev_label = i.final_label
    r = Review(inspection_id=i.id, reviewer_id=user.id, label=body.label, defect_type=body.defect_type.strip(),
               severity=body.severity, comment=body.comment, regions=[list(b) for b in body.regions],
               agrees_with_model=agrees, add_to_feedback=body.add_to_feedback)
    db.add(r)
    i.final_label = body.label
    i.review_status = "reviewed"
    i.reviewed_at = utcnow()
    db.flush()
    fb_id = None
    if body.add_to_feedback and i.status == "ok" and not i.images_purged:
        fb = FeedbackSample(inspection_id=i.id, review_id=r.id, product_key=i.product_key, label=body.label,
                            defect_type=body.defect_type.strip())
        db.add(fb)
        db.flush()
        fb_id = fb.id
    audit.record(db, "review.submitted", user, "inspection", i.id,
                 {"review_id": r.id, "label": body.label, "severity": body.severity, "defect_type": body.defect_type,
                  "model_decision": i.decision, "agrees_with_model": agrees, "previous_final_label": prev_label,
                  "regions": len(body.regions), "feedback_id": fb_id})
    db.commit()
    db.refresh(i)
    return {"review": review_out(r), "inspection": inspection_out(i, detail=True), "feedback_id": fb_id}


@router.get("/reviews")
def list_reviews(limit: int = 100, _: User = Depends(require_any), db: Session = Depends(get_db)):
    rows = db.execute(select(Review).order_by(Review.id.desc()).limit(min(limit, 1000))).scalars()
    return [review_out(r) for r in rows]


@router.get("/feedback")
def list_feedback(status: str | None = None, product_key: str | None = None, _: User = Depends(require_any),
                  db: Session = Depends(get_db)):
    q = select(FeedbackSample).order_by(FeedbackSample.id.desc())
    if status:
        q = q.where(FeedbackSample.status == status)
    if product_key:
        q = q.where(FeedbackSample.product_key == product_key)
    out = []
    for f in db.execute(q.limit(1000)).scalars():
        i = db.get(Inspection, f.inspection_id)
        out.append({"id": f.id, "inspection_id": f.inspection_id, "review_id": f.review_id, "product_key": f.product_key,
                    "label": f.label, "defect_type": f.defect_type, "status": f.status, "dataset_id": f.dataset_id,
                    "created_at": f.created_at.isoformat(), "model_decision": i.decision if i else None,
                    "thumbnail": f"/api/inspections/{f.inspection_id}/image/display" if i and i.display_path else None})
    counts = dict(db.execute(select(FeedbackSample.status, func.count()).group_by(FeedbackSample.status)).all())
    return {"items": out, "counts": counts}


class FeedbackDecision(BaseModel):
    approve: bool


@router.post("/feedback/{feedback_id}/decision")
def feedback_decision(feedback_id: int, body: FeedbackDecision, user: User = Depends(require_admin),
                      db: Session = Depends(get_db)):
    fb = db.get(FeedbackSample, feedback_id)
    if fb is None:
        raise HTTPException(404, "Feedback sample not found")
    try:
        decide_feedback(db, fb, body.approve, user)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    db.commit()
    return {"id": fb.id, "status": fb.status}
