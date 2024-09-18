"""Inspection → decision → review → feedback → approval, plus fail-safe and duplicate handling."""

from __future__ import annotations

import uuid

import numpy as np

from .conftest import png_bytes


def _inspect(client, data: bytes, name="img.png", product="demo-pcb", **form):
    return client.post("/api/inspections", files={"file": (name, data, "image/png")},
                       data={"product_key": product, **form})


def test_products_have_active_models(admin):
    prods = {p["key"]: p for p in admin.get("/api/products").json()}
    assert {"demo-pcb", "demo-metal"} <= set(prods)
    assert prods["demo-pcb"]["active_model_id"] is not None


def test_inspect_normal_and_anomalous(inspector, demo_images):
    normal, anomalous = demo_images
    rn = _inspect(inspector, normal, "normal.png", force="true")
    ra = _inspect(inspector, anomalous, "anomalous.png", force="true")
    assert rn.status_code == 201 and ra.status_code == 201, (rn.text, ra.text)
    n, a = rn.json(), ra.json()
    for r in (n, a):
        assert r["decision"] in ("PASS", "FAIL", "REVIEW")
        assert r["status"] == "ok"
        assert r["score"] is not None and r["model_id"] is not None
        # raw anomaly score and calibrated probability are reported separately
        assert r["calibrated_probability"] is None or 0.0 <= r["calibrated_probability"] <= 1.0
        assert r["reasons"], "every decision must carry human-readable reasons"
    assert a["score"] > n["score"]
    assert a["n_regions"] >= 1 and a["regions"][0]["bbox"]
    # decision policy is applied exactly as documented
    for r in (n, a):
        if r["score"] >= r["fail_threshold"]:
            assert r["decision"] == "FAIL"
        elif r["score"] >= r["pass_threshold"]:
            assert r["decision"] == "REVIEW"
        if r["decision"] == "PASS":
            assert r["score"] < r["pass_threshold"]


def test_inspection_artifacts_and_report(inspector, demo_images):
    _, anomalous = demo_images
    r = _inspect(inspector, anomalous, "art.png", force="true").json()
    iid = r["id"]
    for kind in ("display", "heatmap", "overlay"):
        img = inspector.get(f"/api/inspections/{iid}/image/{kind}")
        assert img.status_code == 200 and img.headers["content-type"].startswith("image/"), kind
    pdf = inspector.get(f"/api/inspections/{iid}/report.pdf")
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
    assert inspector.get(f"/api/inspections/{iid}/image/nope").status_code == 404
    csv = inspector.get("/api/inspections/export.csv")
    assert csv.status_code == 200 and "decision" in csv.text.splitlines()[0]


def test_duplicate_and_idempotency(inspector, demo_images):
    normal, _ = demo_images
    img = bytearray(normal)
    first = _inspect(inspector, bytes(img), "dup.png", force="true").json()
    again = _inspect(inspector, bytes(img), "dup.png").json()
    assert again["duplicate"] is True and again["id"] == first["id"]

    key = str(uuid.uuid4())
    rng = np.random.default_rng(7)
    fresh = png_bytes(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    a = inspector.post("/api/inspections", files={"file": ("k.png", fresh, "image/png")},
                       data={"product_key": "demo-pcb"}, headers={"Idempotency-Key": key}).json()
    b = inspector.post("/api/inspections", files={"file": ("k.png", fresh, "image/png")},
                       data={"product_key": "demo-pcb"}, headers={"Idempotency-Key": key}).json()
    assert b["id"] == a["id"] and b["duplicate"] is True


def test_corrupt_upload_rejected_from_ui(inspector):
    r = _inspect(inspector, b"this is not an image", "bad.png")
    assert r.status_code == 422


def test_corrupt_camera_frame_fails_safe_to_review(inspector):
    r = _inspect(inspector, b"\x89PNG\r\n\x1a\n" + b"\x00" * 50, "frame.png", source="api")
    assert r.status_code == 201
    body = r.json()
    assert body["decision"] == "REVIEW" and body["status"] == "error"
    assert body["needs_review"] is True


def test_blank_frame_is_never_auto_passed(inspector):
    black = png_bytes(np.zeros((256, 256, 3), dtype=np.uint8))
    r = _inspect(inspector, black, "black.png", force="true").json()
    assert r["decision"] != "PASS"
    assert any("quality" in s.lower() or "dark" in s.lower() or "out-of-distribution" in s.lower() or "ood" in s.lower()
               for s in r["reasons"])


def test_oversize_upload_413(inspector):
    big = b"\x00" * (3 * 1024 * 1024)  # DL_MAX_UPLOAD_MB=2 in tests
    assert _inspect(inspector, big, "big.png").status_code == 413


def test_unknown_product_rejected(inspector, demo_images):
    r = _inspect(inspector, demo_images[0], "x.png", product="no-such-product")
    assert r.status_code in (404, 409, 422)


def test_review_feedback_approval_flow(admin, inspector, demo_images):
    _, anomalous = demo_images
    insp = _inspect(inspector, anomalous, "rev.png", force="true").json()
    queue = inspector.get("/api/reviews/queue").json()
    assert "items" in queue

    # defective requires a severity other than "none"
    bad = inspector.post(f"/api/reviews/{insp['id']}", json={"label": "defective", "severity": "none"})
    assert bad.status_code == 422
    r = inspector.post(f"/api/reviews/{insp['id']}", json={
        "label": "defective", "defect_type": "scratch", "severity": "medium", "comment": "confirmed",
        "regions": [[10, 10, 20, 20]], "add_to_feedback": True})
    assert r.status_code == 201, r.text
    assert r.json()["review"]["agrees_with_model"] == (insp["decision"] == "FAIL")

    detail = inspector.get(f"/api/inspections/{insp['id']}").json()
    assert detail["review_status"] == "reviewed"
    # the model's original decision is preserved alongside the human label
    assert detail["decision"] == insp["decision"]

    fb = admin.get("/api/feedback", params={"status": "pending"}).json()["items"]
    mine = [f for f in fb if f["inspection_id"] == insp["id"]]
    assert mine and mine[0]["label"] == "defective"
    assert inspector.post(f"/api/feedback/{mine[0]['id']}/decision", json={"approve": True}).status_code == 403
    ok = admin.post(f"/api/feedback/{mine[0]['id']}/decision", json={"approve": True})
    assert ok.status_code == 200 and ok.json()["status"] == "approved"

    trail = [e["action"] for e in inspector.get(f"/api/inspections/{insp['id']}/audit").json()]
    assert any(a.startswith("inspection.") for a in trail) and any(a.startswith("review.") for a in trail)


def test_review_region_validation(inspector, demo_images):
    insp = _inspect(inspector, demo_images[1], "reg.png", force="true").json()
    r = inspector.post(f"/api/reviews/{insp['id']}", json={"label": "defective", "severity": "low",
                                                           "regions": [[1, 2, 3]]})
    assert r.status_code == 422


def test_history_filters_and_sort(inspector):
    r = inspector.get("/api/inspections", params={"product_key": "demo-pcb", "sort": "-score", "page_size": 50}).json()
    # sorted by normalised score (score / reject threshold) so different models and thresholds are comparable
    scores = [i["normalized_score"] for i in r["items"] if i["normalized_score"] is not None]
    assert scores == sorted(scores, reverse=True)
    for d in ("PASS", "FAIL", "REVIEW"):
        items = inspector.get("/api/inspections", params={"decision": d}).json()["items"]
        assert all(i["decision"] == d for i in items)


def test_dashboard_counts_are_consistent(admin):
    d = admin.get("/api/dashboard").json()
    total = admin.get("/api/inspections", params={"page_size": 1}).json()["total"]
    assert d["total"] == total
    assert d["pass"] + d["fail"] + d["review"] == d["total"]
    assert d["errors"] <= d["review"]  # inspections that could not run are always routed to REVIEW
