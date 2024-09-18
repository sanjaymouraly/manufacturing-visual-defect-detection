"""Report generation: CSV/JSON exports and PDF inspection / period reports."""

from __future__ import annotations

import csv
import io
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sqlalchemy import select

from app.config import get_settings
from app.db import SessionLocal
from app.jobs import JobContext, handler
from app.models import Inspection
from app.services.analytics import full_analytics, overview
from app.services.storage import abs_data_path

CSV_FIELDS = ["id", "created_at", "product_key", "decision", "severity", "score", "normalized_score",
              "calibrated_probability", "fail_threshold", "pass_threshold", "ood_score", "model_name", "algorithm",
              "model_id", "source", "batch_code", "camera_id", "station", "external_ref", "filename", "image_sha256",
              "width", "height", "latency_ms", "status", "error", "review_status", "final_label", "reviewed_at",
              "duplicate_of", "n_regions", "reasons"]
DISCLAIMER = ("Decision support output. Anomaly scores are unitless model outputs, not probabilities unless "
              "explicitly marked as calibrated. Final quality decisions remain with authorised inspectors.")


def inspection_row(i: Inspection) -> dict:
    d = {f: getattr(i, f, None) for f in CSV_FIELDS if hasattr(i, f)}
    d["created_at"] = i.created_at.isoformat() if i.created_at else ""
    d["reviewed_at"] = i.reviewed_at.isoformat() if i.reviewed_at else ""
    d["n_regions"] = len(i.regions or [])
    d["reasons"] = " | ".join(i.reasons or [])
    return d


def to_csv(rows: list[Inspection]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_FIELDS)
    w.writeheader()
    for r in rows:
        row = inspection_row(r)
        # neutralise spreadsheet formula injection
        for k, v in row.items():
            if isinstance(v, str) and v[:1] in ("=", "+", "-", "@"):
                row[k] = "'" + v
        w.writerow(row)
    return buf.getvalue()


def _styles():
    ss = getSampleStyleSheet()
    return ss["Title"], ss["Heading2"], ss["BodyText"], ss["Italic"]


def _table(data: list[list], widths=None) -> Table:
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTSIZE", (0, 0), (-1, -1), 8),
                           ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")])]))
    return t


def _img(rel: str, width: float) -> RLImage | None:
    if not rel:
        return None
    try:
        p = abs_data_path(rel)
    except ValueError:
        return None
    if not p.exists():
        return None
    from PIL import Image

    with Image.open(p) as im:
        w, h = im.size
    return RLImage(str(p), width=width, height=width * h / w)


def inspection_pdf(i: Inspection, llm_summary: str | None = None) -> bytes:
    title, h2, body, ital = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.5 * cm)
    el = [Paragraph(f"Inspection report {i.id}", title), Paragraph(DISCLAIMER, ital), Spacer(1, 6)]
    rows = [["Field", "Value"], ["Timestamp (UTC)", i.created_at.strftime("%Y-%m-%d %H:%M:%S")],
            ["Product", i.product_key], ["Decision", i.decision], ["Severity", i.severity],
            ["Anomaly score (raw)", f"{i.score:.5g}" if i.score is not None else "n/a"],
            ["Normalised score (score / reject thr.)", f"{i.normalized_score:.3f}" if i.normalized_score is not None else "n/a"],
            ["Calibrated P(defect)", f"{i.calibrated_probability:.3f} (Platt, validation split)"
             if i.calibrated_probability is not None else "not calibrated"],
            ["Thresholds (pass / reject)", f"{i.pass_threshold} / {i.fail_threshold}"],
            ["Model", f"{i.model_name} (id {i.model_id}, {i.algorithm})"],
            ["Batch / camera / station", f"{i.batch_code or '-'} / {i.camera_id or '-'} / {i.station or '-'}"],
            ["File / SHA-256", f"{i.filename} / {i.image_sha256[:16]}..."],
            ["Latency", f"{i.latency_ms} ms" if i.latency_ms is not None else "n/a"],
            ["Review status / final label", f"{i.review_status} / {i.final_label or '-'}"]]
    el.append(_table(rows, [6 * cm, 11 * cm]))
    el.append(Spacer(1, 8))
    el.append(Paragraph("Decision reasons", h2))
    for r in i.reasons or []:
        el.append(Paragraph(f"&bull; {r}", body))
    imgs = [x for x in (_img(i.display_path, 8.5 * cm), _img(i.overlay_path, 8.5 * cm)) if x]
    if imgs:
        el.append(Paragraph("Original image and anomaly overlay", h2))
        el.append(Table([imgs]))
    if i.regions:
        el.append(Paragraph("Localised regions (original image pixels)", h2))
        el.append(_table([["#", "bbox [x,y,w,h]", "area %", "max score"]] +
                         [[k + 1, str(r["bbox"]), f"{100 * r['area_fraction']:.2f}", f"{r['max_score']:.4g}"]
                          for k, r in enumerate(i.regions[:15])]))
    if i.reviews:
        el.append(Paragraph("Human reviews (append-only)", h2))
        el.append(_table([["Time", "Reviewer", "Label", "Type", "Severity", "Comment"]] +
                         [[r.created_at.strftime("%Y-%m-%d %H:%M"), r.reviewer.username if r.reviewer else str(r.reviewer_id),
                           r.label, r.defect_type or "-", r.severity, (r.comment or "")[:60]] for r in i.reviews]))
    if llm_summary:
        el.append(Paragraph("AI-generated summary (model-generated text; verify against data above)", h2))
        el.append(Paragraph(llm_summary.replace("\n", "<br/>"), body))
    doc.build(el)
    return buf.getvalue()


@handler("report")
def period_report_job(ctx: JobContext) -> dict:
    p = ctx.params
    days = int(p.get("days", 30))
    product = p.get("product_key") or None
    batch = p.get("batch_code") or None
    title, h2, body, ital = _styles()
    out_dir = get_settings().reports_dir
    fname = f"report-{datetime.now():%Y%m%d-%H%M%S}-{ctx.job_id[:6]}.pdf"
    path = out_dir / fname
    with SessionLocal() as db:
        ov = overview(db, product, days)
        an = full_analytics(db, product, days)
        q = select(Inspection).where(Inspection.duplicate_of.is_(None)).order_by(Inspection.created_at.desc())
        if product:
            q = q.where(Inspection.product_key == product)
        if batch:
            q = q.where(Inspection.batch_code == batch)
        rows = db.execute(q.limit(int(p.get("max_rows", 300)))).scalars().all()
        ctx.progress(0.4, "Building PDF")
        doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=1.2 * cm, rightMargin=1.2 * cm)
        el = [Paragraph("Quality inspection report", title),
              Paragraph(f"Period: last {days} days | Product: {product or 'all'} | Batch: {batch or 'all'} | "
                        f"Generated {datetime.now():%Y-%m-%d %H:%M}", body), Paragraph(DISCLAIMER, ital), Spacer(1, 8),
              Paragraph("Summary", h2),
              _table([["Inspected", "PASS", "FAIL", "REVIEW", "Errors", "Pending reviews", "Anomaly rate"],
                      [ov["total"], ov["pass"], ov["fail"], ov["review"], ov["errors"], ov["pending_reviews"],
                       f"{100 * ov['anomaly_rate']:.1f}%" if ov["anomaly_rate"] is not None else "-"]])]
        mv = an["model_vs_human"]
        if mv:
            el += [Paragraph("Model decisions vs. reviewer ground truth (reviewed items only)", h2),
                   _table([["Outcome", "Count"]] + [[k, v] for k, v in sorted(mv.items())])]
        if an["batches"]:
            el += [Paragraph("Production batch summary", h2),
                   _table([["Batch", "Total", "PASS", "FAIL", "REVIEW", "Fail rate"]] +
                          [[b["batch_code"], b["total"], b["PASS"], b["FAIL"], b["REVIEW"], f"{100 * b['fail_rate']:.1f}%"]
                           for b in an["batches"][:40]])]
        el += [PageBreak(), Paragraph(f"Inspections (latest {len(rows)})", h2),
               _table([["Time", "Product", "Decision", "Sev.", "Score", "Norm.", "Batch", "File", "Review"]] +
                      [[r.created_at.strftime("%m-%d %H:%M"), r.product_key[:14], r.decision, r.severity,
                        f"{r.score:.3g}" if r.score is not None else "-",
                        f"{r.normalized_score:.2f}" if r.normalized_score is not None else "-",
                        (r.batch_code or "-")[:12], (r.filename or "")[:22], r.final_label or r.review_status]
                       for r in rows])]
        doc.build(el)
    ctx.progress(1.0, "Report ready")
    return {"file": fname, "rows": len(rows)}


def report_path(name: str) -> Path:
    base = get_settings().reports_dir.resolve()
    p = (base / Path(name).name).resolve()
    if base not in p.parents:
        raise ValueError("invalid report name")
    return p
