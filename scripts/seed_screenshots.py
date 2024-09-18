"""Seed a running DefectLens instance with real inspection and benchmark activity.

This is a *demonstration* helper used to produce the documentation screenshots. It does not invent
results: every inspection is produced by the configured model through the public API, and every
evaluation is run by the application's normal background jobs on the real dataset. Review outcomes
are deliberately not seeded - the review queue shown in the screenshots contains genuine pending
items awaiting a human.

Typical use (from the repository root, backend virtualenv active, server already running):

    python scripts/seed_screenshots.py --base-url http://127.0.0.1:8010 \
        --product visa-pcb1 --dataset-dir var/datasets/visa-pcb1 \
        --limit 130 --evaluate --robustness

The admin password is read from ``E2E_ADMIN_PASSWORD`` or ``DL_ADMIN_PASSWORD``; otherwise you are
prompted. The script is idempotent for inspections (it sends an Idempotency-Key derived from the
dataset row), but evaluation/robustness runs are appended each time it is called.
"""

from __future__ import annotations

import argparse
import csv
import getpass
import os
import sys
import time
from pathlib import Path

import httpx

CSRF = {"X-Requested-With": "defectlens"}
TERMINAL = {"succeeded", "failed", "cancelled"}


def login(client: httpx.Client, username: str, password: str) -> None:
    r = client.post("/api/auth/login", json={"username": username, "password": password}, headers=CSRF)
    r.raise_for_status()


def choose_test_rows(manifest: Path, limit: int) -> list[dict]:
    with manifest.open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] == "test"]
    normals = sorted((r for r in rows if r["label"] == "normal"), key=lambda r: r["path"])
    anomalies = sorted((r for r in rows if r["label"] == "anomaly"), key=lambda r: r["path"])
    # Interleave classes so batch statistics are not trivially all-good / all-bad.
    out: list[dict] = []
    for i in range(max(len(normals), len(anomalies))):
        if i < len(anomalies):
            out.append(anomalies[i])
        if i < len(normals):
            out.append(normals[i])
    return out[:limit]


def seed_inspections(client: httpx.Client, dataset_dir: Path, product: str, rows: list[dict]) -> dict[str, int]:
    counts = {"PASS": 0, "REVIEW": 0, "FAIL": 0, "error": 0, "duplicate": 0}
    batches = ["PCB1-2026-W40-A", "PCB1-2026-W41-B", "PCB1-2026-W42-C", "PCB1-2026-W43-D"]
    for idx, row in enumerate(rows, 1):
        path = dataset_dir / row["path"]
        batch = batches[min(len(batches) - 1, idx * len(batches) // (len(rows) + 1))]
        r = client.post(
            "/api/inspections",
            files={"file": (path.name, path.read_bytes(), "image/png")},
            data={"product_key": product, "batch_code": batch, "station": "SMT-AOI-1", "source": "upload",
                  "external_ref": Path(row["path"]).stem},
            headers={**CSRF, "Idempotency-Key": f"seed-{product}-{row['path']}"},
            timeout=120,
        )
        if r.status_code >= 400:
            raise RuntimeError(f"inspection failed for {path}: HTTP {r.status_code} {r.text[:200]}")
        d = r.json()
        if d.get("duplicate"):
            counts["duplicate"] += 1
        else:
            key = d.get("decision") if d.get("status") == "ok" else "error"
            counts[key] = counts.get(key, 0) + 1
        if idx % 20 == 0 or idx == len(rows):
            print(f"  inspected {idx}/{len(rows)}  {counts}")
    return counts


def wait_for_job(client: httpx.Client, job_id: str, timeout_s: int = 3600) -> dict:
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        r = client.get(f"/api/jobs/{job_id}", headers=CSRF)
        r.raise_for_status()
        job = r.json()
        msg = f"{job['status']} {round(100 * (job.get('progress') or 0))}% {job.get('message') or ''}"
        if msg != last:
            print(f"  job {job_id}: {msg}")
            last = msg
        if job["status"] in TERMINAL:
            if job["status"] != "succeeded":
                raise RuntimeError(f"job {job_id} {job['status']}: {job.get('error')}")
            return job
        time.sleep(2)
    raise TimeoutError(f"job {job_id} did not finish within {timeout_s}s")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=os.environ.get("E2E_BASE_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--username", default=os.environ.get("E2E_ADMIN_USER", "admin"))
    ap.add_argument("--password", default=os.environ.get("E2E_ADMIN_PASSWORD") or os.environ.get("DL_ADMIN_PASSWORD"))
    ap.add_argument("--product", default="visa-pcb1")
    ap.add_argument("--dataset-dir", type=Path, default=Path("var/datasets/visa-pcb1"))
    ap.add_argument("--limit", type=int, default=130, help="number of test images to inspect (0 = all)")
    ap.add_argument("--evaluate", action="store_true", help="run a Model lab benchmark on all models of the product")
    ap.add_argument("--robustness", action="store_true", help="run the 19 x 3 robustness benchmark on PatchCore")
    ap.add_argument("--robustness-samples", type=int, default=40)
    args = ap.parse_args(argv)
    if not args.password:
        args.password = getpass.getpass(f"Password for {args.username}: ")

    manifest = args.dataset_dir / "manifest.csv"
    if not manifest.exists():
        sys.exit(f"{manifest} not found - pass --dataset-dir pointing at a prepared dataset")

    with httpx.Client(base_url=args.base_url, timeout=60) as client:
        login(client, args.username, args.password)
        print(f"Signed in to {args.base_url}")

        datasets = client.get("/api/datasets", headers=CSRF).json()
        ds = next((d for d in datasets if d["name"] == args.product), None)
        if ds is None:
            sys.exit(f"Dataset '{args.product}' is not registered - run: defectlens register-datasets")
        models = [m for m in client.get("/api/models", headers=CSRF).json() if m["product_key"] == args.product]
        if not models:
            sys.exit(f"No models for '{args.product}' - train them first")
        products = client.get("/api/products", headers=CSRF).json()
        product = next((p for p in products if p["key"] == args.product), None)
        if product is None or product["active_model_id"] is None:
            sys.exit(f"Product '{args.product}' has no active model - activate one in Model lab")

        rows = choose_test_rows(manifest, args.limit)
        print(f"Inspecting {len(rows)} held-out test images as {args.product} ...")
        counts = seed_inspections(client, args.dataset_dir, args.product, rows)
        print("Inspection decisions:", counts)

        if args.evaluate:
            payload = {"dataset_id": ds["id"], "model_ids": [m["id"] for m in models], "split": "test"}
            r = client.post("/api/models/evaluate", json=payload, headers=CSRF)
            r.raise_for_status()
            print("Benchmark job:", r.json()["id"])
            wait_for_job(client, r.json()["id"])

        if args.robustness:
            patchcore = next((m for m in models if m["algorithm"] == "patchcore"), None)
            if patchcore is None:
                sys.exit("No PatchCore model available for the robustness run")
            payload = {"model_id": patchcore["id"], "dataset_id": ds["id"], "degradations": [],
                       "severities": [1, 3, 5], "max_samples": args.robustness_samples}
            r = client.post("/api/robustness", json=payload, headers=CSRF)
            r.raise_for_status()
            print("Robustness job:", r.json()["id"])
            wait_for_job(client, r.json()["id"], timeout_s=7200)

    print("Done. Refresh the UI; the documentation screenshots can now be captured.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
