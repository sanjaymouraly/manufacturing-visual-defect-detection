"""Held-out benchmarking and model comparison on identical test data."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from defectml.data.manifest import Dataset
from defectml.evaluation.metrics import (
    bootstrap_auroc_ci,
    calibration_metrics,
    pixel_metrics,
    platt_apply,
    ranking_metrics,
    threshold_metrics,
    wilson_interval,
)
from defectml.models import load_model
from defectml.training import peak_memory_mb, predict_samples


def policy_outcomes(y: np.ndarray, s: np.ndarray, fail_t: float, pass_t: float) -> dict:
    """Three-way outcome rates of the PASS / REVIEW / FAIL policy (score-only, before quality/OOD)."""
    dec = np.where(s >= fail_t, "FAIL", np.where(s < pass_t, "PASS", "REVIEW"))
    out: dict = {}
    for label, name in ((0, "normal"), (1, "anomalous")):
        sel = y == label
        n = int(sel.sum())
        out[name] = {d: int((dec[sel] == d).sum()) for d in ("PASS", "REVIEW", "FAIL")} | {"n": n}
    n_anom = out["anomalous"]["n"]
    escaped = out["anomalous"]["PASS"]
    out["auto_pass_miss_rate"] = escaped / n_anom if n_anom else None
    out["auto_pass_miss_rate_ci95"] = wilson_interval(escaped, n_anom)
    out["review_rate"] = float((dec == "REVIEW").mean()) if len(dec) else None
    return out


def evaluate_model(model_dir: Path, ds: Dataset, split: str = "test", max_samples: int | None = None,
                   failure_examples: int = 6, progress=None, cancel=None) -> dict:
    model, meta = load_model(model_dir)
    samples = ds.split(split)
    if max_samples and len(samples) > max_samples:
        rng = np.random.default_rng(0)
        samples = [samples[i] for i in sorted(rng.choice(len(samples), max_samples, replace=False))]
    if not samples:
        raise ValueError(f"Split '{split}' is empty")
    t0 = time.perf_counter()
    r = predict_samples(model, ds, samples, with_masks=ds.info.masks_available, progress=progress, cancel=cancel)
    wall = time.perf_counter() - t0
    y, s = r["y"], r["scores"]
    th = meta["thresholds"]
    res: dict = {
        "model": meta["name"], "algorithm": meta["algorithm"], "product": meta["product"],
        "dataset": ds.info.name, "synthetic_data": ds.info.synthetic, "split": split, "n": len(samples),
        "n_anomalous": int(y.sum()), "n_normal": int((y == 0).sum()),
        "thresholds": th,
        "ranking": ranking_metrics(y, s),
        "auroc_ci95": bootstrap_auroc_ci(y, s),
        "at_fail_threshold": threshold_metrics(y, s, th["fail_threshold"]),
        "policy": policy_outcomes(y, s, th["fail_threshold"], th["pass_threshold"]),
        "latency_ms_per_image": round(r["latency_ms_per_image"], 2),
        "throughput_img_per_s": round(len(samples) / wall, 2) if wall > 0 else None,
        "peak_memory_mb": peak_memory_mb(),
    }
    if meta.get("calibration"):
        prob = platt_apply(meta["calibration"], s)
        res["calibration_test"] = calibration_metrics(y, prob)
    if ds.info.masks_available and r["masks"]:
        res["pixel"] = pixel_metrics(r["masks"], r["maps"], th["seg_threshold"])
    if ds.info.defect_types_annotated:
        per: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
        for smp, sc in zip(samples, s, strict=True):
            if smp.is_anomaly:
                for t in (smp.defect_type or "unspecified").split("|"):
                    per[t][1] += 1
                    per[t][0] += int(sc >= th["fail_threshold"])
        res["per_defect_type_recall"] = {
            t: {"recall": k / n, "detected": k, "n": n, "ci95": wilson_interval(k, n)} for t, (k, n) in sorted(per.items())
        }
    # failure examples: highest-scoring normals (false alarms) and lowest-scoring anomalies (misses)
    order = np.argsort(s)
    fn = [i for i in order if y[i] == 1 and s[i] < th["fail_threshold"]][:failure_examples]
    fp = [i for i in order[::-1] if y[i] == 0 and s[i] >= th["fail_threshold"]][:failure_examples]
    res["failure_examples"] = {
        "missed_defects": [{"path": samples[i].path, "score": float(s[i]), "defect_type": samples[i].defect_type}
                           for i in fn],
        "false_alarms": [{"path": samples[i].path, "score": float(s[i])} for i in fp],
    }
    res["scores"] = {"y": y.tolist(), "score": [round(float(v), 5) for v in s]}
    return res


def compare(results: list[dict], out_dir: Path | None = None) -> dict:
    rows = []
    for r in results:
        a = r["at_fail_threshold"]
        rows.append({
            "model": r["model"], "algorithm": r["algorithm"], "dataset": r["dataset"], "n": r["n"],
            "auroc": r["ranking"]["auroc"], "auroc_ci95": r["auroc_ci95"], "ap": r["ranking"]["average_precision"],
            "f1": a["f1"], "precision": a["precision"], "recall": a["recall"], "balanced_accuracy": a["balanced_accuracy"],
            "miss_rate": a["miss_rate"], "false_rejection_rate": a["false_rejection_rate"],
            "recall_at_fpr_0.05": r["ranking"]["recall_at_fpr"].get("0.05"),
            "auto_pass_miss_rate": r["policy"]["auto_pass_miss_rate"], "review_rate": r["policy"]["review_rate"],
            "pixel_auroc": (r.get("pixel") or {}).get("pixel_auroc"), "aupro": (r.get("pixel") or {}).get("aupro_0.3"),
            "dice": (r.get("pixel") or {}).get("dice"), "ece": (r.get("calibration_test") or {}).get("ece"),
            "latency_ms": r["latency_ms_per_image"], "throughput": r["throughput_img_per_s"],
        })
    summary = {"rows": rows, "results": results}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "comparison.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
        plot_comparison(results, out_dir)
    return summary


def plot_comparison(results: list[dict], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for r in results:
        rc = r["ranking"].get("roc_curve")
        if rc:
            axes[0].plot(rc["fpr"], rc["tpr"], label=f"{r['algorithm']} (AUROC {r['ranking']['auroc']:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title(f"ROC - {results[0]['dataset']} ({results[0]['split']}, n={results[0]['n']})")
    axes[0].legend(fontsize=8)
    names = [r["algorithm"] for r in results]
    miss = [r["at_fail_threshold"]["miss_rate"] or 0 for r in results]
    frr = [r["at_fail_threshold"]["false_rejection_rate"] or 0 for r in results]
    x = np.arange(len(names))
    axes[1].bar(x - 0.2, miss, 0.4, label="Miss rate (defects passed)")
    axes[1].bar(x + 0.2, frr, 0.4, label="False-rejection rate")
    axes[1].set_xticks(x, names)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Operating point (validation-selected threshold)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison.png", dpi=110)
    plt.close(fig)
