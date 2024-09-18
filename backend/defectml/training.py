"""Model training with validation-only threshold selection and calibration.

The held-out ``test`` split is never touched here. Thresholds, pixel threshold, OOD threshold and
Platt calibration are fitted on the ``val`` split only.
"""

from __future__ import annotations

import platform
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from defectml.data.manifest import Dataset, Sample
from defectml.evaluation.metrics import (
    platt_fit,
    ranking_metrics,
    select_pixel_threshold,
    select_threshold_youden,
    threshold_for_recall,
    threshold_metrics,
)
from defectml.imaging import load_image, load_mask
from defectml.models import build_model, save_model
from defectml.models.base import AnomalyModel, CancelFn, ProgressFn


def peak_memory_mb() -> float | None:
    try:
        import resource

        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # type: ignore[attr-defined]  # POSIX only
        return round(r / 1024 if platform.system() != "Darwin" else r / (1024 * 1024), 1)
    except Exception:
        return None


def iter_images(ds: Dataset, samples: list[Sample]) -> Iterator[np.ndarray]:
    for s in samples:
        yield load_image(ds.abspath(s.path))


def predict_samples(model: AnomalyModel, ds: Dataset, samples: list[Sample], batch_size: int = 16,
                    with_masks: bool = True, progress: ProgressFn | None = None, cancel: CancelFn | None = None,
                    transform=None) -> dict:
    """Run the model on samples. ``transform(img, mask) -> (img, mask)`` applies degradations."""
    scores, maps, masks, oods = [], [], [], []
    t_total = 0.0
    for i in range(0, len(samples), batch_size):
        if cancel is not None and cancel():
            raise RuntimeError("cancelled")
        chunk = samples[i:i + batch_size]
        imgs = []
        for s in chunk:
            img = load_image(ds.abspath(s.path))
            mask = load_mask(ds.abspath(s.mask) if s.mask else None, img.shape[:2]) if with_masks else None
            if transform is not None:
                img, mask = transform(img, mask)
            imgs.append(img)
            if with_masks and mask is not None:
                import cv2

                size = model.config["input_size"]
                masks.append(cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST))
        t0 = time.perf_counter()
        preds = model.predict_batch(imgs)
        t_total += time.perf_counter() - t0
        for p in preds:
            scores.append(p.score)
            maps.append(p.anomaly_map)
            oods.append(p.ood_score if p.ood_score is not None else np.nan)
        if progress:
            progress(min(1.0, (i + len(chunk)) / len(samples)), f"Predicted {i + len(chunk)}/{len(samples)}")
    return {
        "y": np.array([s.y for s in samples]),
        "scores": np.array(scores, np.float64),
        "maps": maps,
        "masks": masks,
        "ood": np.array(oods, np.float64),
        "latency_ms_per_image": 1000 * t_total / max(len(samples), 1),
    }


def choose_thresholds(y: np.ndarray, scores: np.ndarray, target_recall: float = 0.98,
                      min_review_margin: float = 0.25) -> dict:
    """fail = Youden-optimal (balanced accuracy, prevalence-independent); pass = recall-targeted,
    and at least ``min_review_margin`` below fail.

    The margin guarantees a REVIEW band even when the validation split is perfectly separable
    (validation sets are small; perfect separation there does not imply it in production).
    """
    normal = scores[y == 0]
    if (y == 1).sum() >= 1 and (y == 0).sum() >= 1:
        fail = select_threshold_youden(y, scores)
        pass_t = min(fail, threshold_for_recall(y, scores, target_recall))
        method = f"fail=Youden-optimal (max TPR-FPR) on val; pass=highest threshold with val recall>={target_recall}"
    else:
        fail = float(np.quantile(normal, 0.995))
        pass_t = float(np.quantile(normal, 0.95))
        method = "no anomalous validation samples: fail=p99.5, pass=p95 of normal val scores"
    lo = float(np.min(scores))
    margin_t = fail - min_review_margin * (fail - lo) if fail > lo else fail
    pass_t = min(pass_t, fail, margin_t)
    return {"fail_threshold": float(fail), "pass_threshold": float(pass_t),
            "method": method + f"; review band >= {min_review_margin:.0%} of (fail - min val score)"}


def train_on_dataset(ds: Dataset, algorithm: str, out_dir: Path, *, name: str | None = None,
                     config: dict | None = None, target_recall: float = 0.98, max_train_images: int | None = None,
                     seed: int = 0, progress: ProgressFn | None = None, cancel: CancelFn | None = None) -> dict:
    def p(frac: float, msg: str) -> None:
        if progress:
            progress(frac, msg)

    train = [s for s in ds.split("train") if not s.is_anomaly]
    val = ds.split("val")
    if len(train) < 2:
        raise ValueError("Training split needs at least 2 normal images")
    if not val:
        raise ValueError("Validation split is empty - cannot select thresholds")
    if max_train_images and len(train) > max_train_images:
        rng = np.random.default_rng(seed)
        train = [train[i] for i in sorted(rng.choice(len(train), max_train_images, replace=False))]
    cfg = dict(config or {})
    cfg.setdefault("seed", seed)
    model = build_model(algorithm, **cfg)
    t0 = time.time()
    model.fit(iter_images(ds, train), len(train), progress=lambda f, m: p(0.05 + 0.7 * f, m), cancel=cancel)
    fit_seconds = time.time() - t0
    p(0.78, "Scoring validation split")
    vr = predict_samples(model, ds, val, with_masks=ds.info.masks_available,
                         progress=lambda f, m: p(0.78 + 0.17 * f, "Validation: " + m), cancel=cancel)
    y, s = vr["y"], vr["scores"]
    th = choose_thresholds(y, s, target_recall)
    seg_t = None
    if ds.info.masks_available and vr["masks"] and (y == 1).any():
        seg_t = select_pixel_threshold(vr["masks"], vr["maps"])
    th["seg_threshold"] = float(seg_t) if seg_t is not None else th["fail_threshold"]
    normal_ood = vr["ood"][(y == 0) & ~np.isnan(vr["ood"])]
    th["ood_threshold"] = float(np.quantile(normal_ood, 0.995) * 1.1) if len(normal_ood) else None
    normal_maps = [m for m, yy in zip(vr["maps"], y, strict=True) if yy == 0]
    map_low = float(np.median([np.median(m) for m in normal_maps])) if normal_maps else float(np.min(s))
    th["map_low"] = map_low
    th["map_high"] = float(max(th["seg_threshold"] * 1.5, map_low + 1e-3))
    calibration = None
    if (y == 1).sum() >= 5 and (y == 0).sum() >= 5:
        calibration = {**platt_fit(y, s), "fitted_on": "val", "n": len(y)}
    val_metrics = {
        "split": "val",
        "n": len(y),
        "ranking": {k: v for k, v in ranking_metrics(y, s).items() if k != "roc_curve"},
        "at_fail_threshold": threshold_metrics(y, s, th["fail_threshold"]),
        "at_pass_threshold": threshold_metrics(y, s, th["pass_threshold"]),
        "latency_ms_per_image": round(vr["latency_ms_per_image"], 2),
    }
    import sklearn

    versions = {"numpy": np.__version__, "scikit-learn": sklearn.__version__, "python": platform.python_version()}
    try:
        import torch

        versions["torch"] = torch.__version__
    except Exception:  # noqa: S110
        pass
    meta = {
        "name": name or f"{ds.info.product}-{algorithm}",
        "product": ds.info.product,
        "dataset": {"name": ds.info.name, "version": ds.info.version, "manifest_sha256": ds.info.manifest_sha256,
                    "license": ds.info.license, "synthetic": ds.info.synthetic, "source": ds.info.source},
        "thresholds": th,
        "calibration": calibration,
        "validation_metrics": val_metrics,
        "training": {"n_train_images": len(train), "fit_seconds": round(fit_seconds, 2), "seed": seed,
                     "target_recall": target_recall, "peak_memory_mb": peak_memory_mb(),
                     "device": model.config.get("device", "cpu"), "platform": platform.platform()},
        "library_versions": versions,
        "created_at": datetime.now(UTC).isoformat(),
        "supports_localization": model.supports_localization,
    }
    full = save_model(model, out_dir, meta)
    p(1.0, "Model saved")
    return full
