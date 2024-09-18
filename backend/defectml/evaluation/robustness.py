"""Robustness test bench and lighting domain-shift experiment.

Thresholds are those selected on the validation split at training time; the robustness bench
reports how clean-data operating points degrade. Nothing here re-tunes models on test data.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from defectml.data.manifest import Dataset
from defectml.evaluation import degradations as D
from defectml.evaluation.metrics import ranking_metrics, threshold_metrics
from defectml.imaging import load_image, load_mask
from defectml.models import load_model
from defectml.training import predict_samples


def _subset(samples, max_samples, seed=0):
    if max_samples and len(samples) > max_samples:
        # stratified subset keeping class balance
        rng = np.random.default_rng(seed)
        pos = [s for s in samples if s.is_anomaly]
        neg = [s for s in samples if not s.is_anomaly]
        k_pos = max(1, round(max_samples * len(pos) / len(samples)))
        pos = [pos[i] for i in sorted(rng.choice(len(pos), min(k_pos, len(pos)), replace=False))]
        neg = [neg[i] for i in sorted(rng.choice(len(neg), min(max_samples - len(pos), len(neg)), replace=False))]
        return pos + neg
    return samples


def _summ(y, s, th) -> dict:
    t = threshold_metrics(y, s, th["fail_threshold"])
    r = ranking_metrics(y, s)
    return {"auroc": r["auroc"], "ap": r["average_precision"], "recall": t["recall"], "miss_rate": t["miss_rate"],
            "false_rejection_rate": t["false_rejection_rate"], "f1": t["f1"]}


def run_robustness(model_dir: Path, ds: Dataset, degradations: list[str] | None = None,
                   severities: list[int] | None = None, split: str = "test", max_samples: int | None = 120,
                   out_dir: Path | None = None, progress=None, cancel=None, n_examples: int = 4) -> dict:
    model, meta = load_model(model_dir)
    th = meta["thresholds"]
    degradations = degradations or list(D.DEGRADATIONS)
    severities = severities or [1, 3, 5]
    for d in degradations:
        if d not in D.DEGRADATIONS:
            raise ValueError(f"Unknown degradation {d}")
    samples = _subset(ds.split(split), max_samples)
    if not samples:
        raise ValueError("No samples in split")
    total_steps = 1 + len(degradations) * len(severities)
    step = 0
    t0 = time.time()

    def prog(msg):
        if progress:
            progress(step / total_steps, msg)

    prog("Clean baseline")
    clean = predict_samples(model, ds, samples, with_masks=False, cancel=cancel)
    y = clean["y"]
    clean_m = _summ(y, clean["scores"], th)
    clean_pred = clean["scores"] >= th["fail_threshold"]
    step += 1
    rows = []
    flips: list[dict] = []
    for d in degradations:
        for sev in severities:
            if cancel is not None and cancel():
                raise RuntimeError("cancelled")
            prog(f"{d} severity {sev}")

            def tf(img, mask, _d=d, _sev=sev):
                seed = int(img[::37, ::41].sum()) % (2**31)  # deterministic per image
                return D.apply(_d, img, mask, _sev, seed=seed)

            r = predict_samples(model, ds, samples, with_masks=False, transform=tf, cancel=cancel)
            m = _summ(y, r["scores"], th)
            pred = r["scores"] >= th["fail_threshold"]
            changed = np.where(pred != clean_pred)[0]
            m.update({"degradation": d, "severity": sev, "decision_flip_rate": float(len(changed) / len(y)),
                      "mean_score_shift": float(np.mean(r["scores"] - clean["scores"]))})
            rows.append(m)
            for i in changed[:2]:
                if len(flips) < 200:
                    flips.append({"path": samples[i].path, "degradation": d, "severity": sev,
                                  "label": samples[i].label, "clean_score": float(clean["scores"][i]),
                                  "degraded_score": float(r["scores"][i])})
            step += 1
    auroc_ratios = [r["auroc"] / clean_m["auroc"] for r in rows if r["auroc"] is not None and clean_m["auroc"]]
    worst = sorted(rows, key=lambda r: (r["auroc"] if r["auroc"] is not None else 1.0))[:5]
    by_deg = {}
    for d in degradations:
        rr = [r for r in rows if r["degradation"] == d and r["auroc"] is not None]
        if rr and clean_m["auroc"]:
            by_deg[d] = round(float(np.mean([r["auroc"] for r in rr]) / clean_m["auroc"]), 4)
    result = {
        "model": meta["name"], "algorithm": meta["algorithm"], "dataset": ds.info.name,
        "synthetic_data": ds.info.synthetic, "split": split, "n": len(samples), "severities": severities,
        "thresholds": {"fail_threshold": th["fail_threshold"]},
        "clean": clean_m, "rows": rows,
        "robustness_score": round(float(np.mean(auroc_ratios)), 4) if auroc_ratios else None,
        "robustness_by_degradation": by_deg,
        "worst_conditions": worst,
        "decision_flips": flips[:50],
        "seconds": round(time.time() - t0, 1),
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "robustness.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
        plot_robustness(result, out_dir / "robustness.png")
        if flips:
            render_failure_examples(ds, flips[:n_examples], out_dir / "failure_examples.jpg")
    if progress:
        progress(1.0, "Robustness run complete")
    return result


def plot_robustness(result: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    degs = sorted({r["degradation"] for r in result["rows"]})
    sevs = result["severities"]
    grid = np.full((len(degs), len(sevs)), np.nan)
    for r in result["rows"]:
        if r["auroc"] is not None:
            grid[degs.index(r["degradation"]), sevs.index(r["severity"])] = r["auroc"]
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, 0.32 * len(degs) + 1)))
    im = axes[0].imshow(grid, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    axes[0].set_yticks(range(len(degs)), degs, fontsize=8)
    axes[0].set_xticks(range(len(sevs)), [f"sev {s}" for s in sevs])
    for i in range(len(degs)):
        for j in range(len(sevs)):
            if not np.isnan(grid[i, j]):
                axes[0].text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=7)
    axes[0].set_title(f"AUROC under degradation (clean {result['clean']['auroc']:.3f})")
    fig.colorbar(im, ax=axes[0], fraction=0.04)
    for d in degs:
        rr = sorted([r for r in result["rows"] if r["degradation"] == d], key=lambda r: r["severity"])
        axes[1].plot([0] + [r["severity"] for r in rr],
                     [result["clean"]["false_rejection_rate"] or 0] + [r["false_rejection_rate"] or 0 for r in rr],
                     marker="o", ms=3, lw=1, label=d)
    axes[1].set_xlabel("severity (0 = clean)")
    axes[1].set_ylabel("false-rejection rate @ val threshold")
    axes[1].set_title(f"{result['algorithm']} - {result['dataset']}")
    axes[1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)


def render_failure_examples(ds: Dataset, flips: list[dict], out: Path, tile: int = 180) -> None:
    tiles = []
    for f in flips:
        img = load_image(ds.abspath(f["path"]))
        seed = int(img[::37, ::41].sum()) % (2**31)
        deg, _ = D.apply(f["degradation"], img, None, f["severity"], seed=seed)
        a = cv2.resize(img, (tile, tile))
        b = cv2.resize(deg, (tile, tile))
        pair = np.concatenate([a, b], axis=1)
        cv2.putText(pair, f"{f['label']} {f['clean_score']:.2f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(pair, f"{f['degradation']} s{f['severity']} {f['degraded_score']:.2f}", (tile + 4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        tiles.append(pair)
    cv2.imwrite(str(out), cv2.cvtColor(np.concatenate(tiles, axis=0), cv2.COLOR_RGB2BGR))


def lighting_shift(img: np.ndarray, mask, seed: int = 0):
    """A fixed 'new lighting installation' shift: warmer, dimmer, with a lighting gradient."""
    rng = np.random.default_rng(seed)
    img, mask = D.color_temperature(img, mask, 3, np.random.default_rng(1))
    img, mask = D.brightness(img, mask, 2, np.random.default_rng(3))  # deterministic direction
    img, mask = D.uneven_lighting(img, mask, 3, rng)
    return img, mask


def domain_shift_experiment(model_dir: Path, ds: Dataset, out_dir: Path | None = None, max_samples: int | None = 150,
                            n_recal: int = 20) -> dict:
    """Simulated capture-condition shift (VisA has no session/camera metadata).

    1. evaluate on clean test data with the validation threshold
    2. evaluate on shifted test data with the same threshold (deployment without adaptation)
    3. re-estimate the acceptance thresholds from ``n_recal`` shifted NORMAL validation images
       (a realistic site-acceptance procedure needing no defect samples) and re-evaluate
    """
    model, meta = load_model(model_dir)
    th = meta["thresholds"]
    test = _subset(ds.split("test"), max_samples)
    val_norm = [s for s in ds.split("val") if not s.is_anomaly][:n_recal]

    def shift(img, mask):
        return lighting_shift(img, mask, seed=int(img[::37, ::41].sum()) % (2**31))

    clean = predict_samples(model, ds, test, with_masks=False)
    shifted = predict_samples(model, ds, test, with_masks=False, transform=shift)
    recal = predict_samples(model, ds, val_norm, with_masks=False, transform=shift)
    # preserve the val-time ratio between fail threshold and the normal-score distribution
    clean_val = predict_samples(model, ds, val_norm, with_masks=False)
    ratio = th["fail_threshold"] / max(float(np.quantile(clean_val["scores"], 0.95)), 1e-9)
    new_fail = float(np.quantile(recal["scores"], 0.95) * ratio)
    y = clean["y"]
    res = {
        "model": meta["name"], "algorithm": meta["algorithm"], "dataset": ds.info.name,
        "shift": "simulated lighting change (warmer colour temperature, -20% brightness, lighting gradient)",
        "n_test": len(test), "n_recalibration_normals": len(val_norm),
        "clean": _summ(y, clean["scores"], th),
        "shifted_original_threshold": _summ(y, shifted["scores"], th),
        "shifted_recalibrated_threshold": _summ(y, shifted["scores"], {"fail_threshold": new_fail}),
        "original_fail_threshold": th["fail_threshold"], "recalibrated_fail_threshold": new_fail,
        "mean_ood_clean": float(np.nanmean(clean["ood"])), "mean_ood_shifted": float(np.nanmean(shifted["ood"])),
        "ood_flag_rate_shifted": float(np.nanmean(shifted["ood"] > th["ood_threshold"])) if th.get("ood_threshold") else None,
        "ood_flag_rate_clean": float(np.nanmean(clean["ood"] > th["ood_threshold"])) if th.get("ood_threshold") else None,
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "domain_shift.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    return res


def degrade_preview(img: np.ndarray, name: str, severity: int, seed: int = 0) -> np.ndarray:
    out, _ = D.apply(name, img, None, severity, seed=seed)
    return out


__all__ = ["domain_shift_experiment", "load_mask", "run_robustness"]
