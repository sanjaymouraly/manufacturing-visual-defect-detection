"""Evaluation metrics for anomaly detection and localisation.

Conventions: label 1 = anomalous/defective (positive class), 0 = normal. Scores: higher = more
anomalous. "Miss rate" = FN / (TP + FN) (defects passed as good). "False-rejection rate" =
FP / (FP + TN) (good parts rejected).
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def _safe_div(a: float, b: float) -> float | None:
    return float(a / b) if b else None


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def confusion(y: np.ndarray, pred: np.ndarray) -> dict:
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    return {
        "tp": int(((y == 1) & (pred == 1)).sum()),
        "fp": int(((y == 0) & (pred == 1)).sum()),
        "tn": int(((y == 0) & (pred == 0)).sum()),
        "fn": int(((y == 1) & (pred == 0)).sum()),
    }


def threshold_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (np.asarray(scores) >= threshold).astype(int)
    c = confusion(y, pred)
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    bal = None if recall is None or specificity is None else (recall + specificity) / 2
    return {
        "threshold": float(threshold),
        "confusion": c,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "balanced_accuracy": bal,
        "accuracy": _safe_div(tp + tn, tp + tn + fp + fn),
        "miss_rate": _safe_div(fn, tp + fn),
        "false_rejection_rate": _safe_div(fp, fp + tn),
        "recall_ci95": wilson_interval(tp, tp + fn),
        "false_rejection_rate_ci95": wilson_interval(fp, fp + tn),
        "n_anomalous": tp + fn,
        "n_normal": tn + fp,
    }


def ranking_metrics(y: np.ndarray, scores: np.ndarray) -> dict:
    y = np.asarray(y).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    if len(np.unique(y)) < 2:
        return {"auroc": None, "average_precision": None, "recall_at_fpr": {}}
    fpr, tpr, _ = roc_curve(y, s)
    at = {}
    for target in (0.01, 0.05, 0.10):
        ok = fpr <= target + 1e-12
        at[f"{target:.2f}"] = float(tpr[ok].max()) if ok.any() else 0.0
    return {
        "auroc": float(roc_auc_score(y, s)),
        "average_precision": float(average_precision_score(y, s)),
        "recall_at_fpr": at,
        "roc_curve": {"fpr": [round(float(v), 4) for v in fpr[:: max(1, len(fpr) // 100)]],
                      "tpr": [round(float(v), 4) for v in tpr[:: max(1, len(tpr) // 100)]]},
    }


def bootstrap_auroc_ci(y: np.ndarray, scores: np.ndarray, n_boot: int = 500, seed: int = 0) -> tuple | None:
    y = np.asarray(y).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    if len(np.unique(y)) < 2:
        return None
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])  # stratified
        vals.append(roc_auc_score(y[idx], s[idx]))
    return (round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4))


def _candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    u = np.unique(scores)
    if len(u) == 1:
        return u
    mids = (u[:-1] + u[1:]) / 2
    return np.concatenate([[u[0] - 1e-6], mids, [u[-1] + 1e-6]])


def select_threshold_f1(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    best_t, best_f1 = float(np.median(s)), -1.0
    for t in _candidate_thresholds(s):
        pred = s >= t
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if f1 > best_f1 + 1e-12:
            best_f1, best_t = f1, float(t)
    return best_t


def select_threshold_youden(y: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximising Youden's J = TPR - FPR (= 2 * balanced accuracy - 1).

    Unlike F1, J does not depend on the anomaly prevalence of the validation split. Benchmark
    validation splits are often 50%+ anomalous while production lines are typically <5%; F1 on
    such a split rewards "reject almost everything". Ties resolve to the HIGHER threshold (fewer
    false rejects at equal J)."""
    y = np.asarray(y).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if not n_pos or not n_neg:
        raise ValueError("Youden's J needs both classes")
    best_t, best_j = float(np.median(s)), -2.0
    for t in _candidate_thresholds(s):
        pred = s >= t
        j = (pred & (y == 1)).sum() / n_pos - (pred & (y == 0)).sum() / n_neg
        if j >= best_j - 1e-12:
            best_j, best_t = float(j), float(t)
    return best_t


def threshold_for_recall(y: np.ndarray, scores: np.ndarray, target_recall: float) -> float:
    """Largest threshold that still achieves ``target_recall`` on the given (validation) data."""
    y = np.asarray(y).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    pos = np.sort(s[y == 1])
    if len(pos) == 0:
        raise ValueError("No positive samples")
    k = math.floor((1 - target_recall) * len(pos) + 1e-9)  # number of positives allowed below
    k = min(max(k, 0), len(pos) - 1)
    return float(pos[k] - 1e-6)


def pixel_metrics(masks: list[np.ndarray], maps: list[np.ndarray], threshold: float,
                  max_fpr: float = 0.3, n_thresholds: int = 40, region_min_overlap: float = 0.3) -> dict:
    """Pixel AUROC/AP, IoU/Dice at ``threshold``, region recall and AUPRO (normalised, FPR<=max_fpr).

    Normal images (empty masks) are included, following the VisA evaluation protocol.
    """
    gt = np.concatenate([m.reshape(-1) for m in masks]).astype(np.uint8)
    sc = np.concatenate([a.reshape(-1) for a in maps]).astype(np.float32)
    out: dict = {"pixels": len(gt), "defect_pixel_fraction": float(gt.mean()) if len(gt) else 0.0}
    if np.min(gt) == np.max(gt):
        out.update({"pixel_auroc": None, "pixel_ap": None})
    else:
        out["pixel_auroc"] = float(roc_auc_score(gt, sc))
        out["pixel_ap"] = float(average_precision_score(gt, sc))
    pred = sc >= threshold
    inter = int((pred & (gt == 1)).sum())
    union = int((pred | (gt == 1)).sum())
    out["iou"] = _safe_div(inter, union)
    out["dice"] = _safe_div(2 * inter, int(pred.sum()) + int(gt.sum()))
    # region-level metrics
    regions = []
    for m, a in zip(masks, maps, strict=True):
        n, lab = cv2.connectedComponents(m.astype(np.uint8))
        for r in range(1, n):
            regions.append(a[lab == r])
    if regions:
        detected = sum(1 for r in regions if (r >= threshold).mean() >= region_min_overlap)
        out["region_recall"] = detected / len(regions)
        out["n_regions"] = len(regions)
        normal_scores = sc[gt == 0]
        if len(normal_scores):
            qs = np.quantile(normal_scores, np.linspace(1.0, 1 - max_fpr, n_thresholds))
            fprs, pros = [], []
            for t in qs:
                fprs.append(float((normal_scores >= t).mean()))
                pros.append(float(np.mean([(r >= t).mean() for r in regions])))
            fprs_a, pros_a = np.array(fprs), np.array(pros)
            order = np.argsort(fprs_a)
            fprs_a, pros_a = fprs_a[order], pros_a[order]
            keep = fprs_a <= max_fpr
            if keep.sum() >= 2:
                trapz = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]  # numpy<2 fallback
                out["aupro_0.3"] = float(trapz(pros_a[keep], fprs_a[keep]) / max_fpr)
    return out


def select_pixel_threshold(masks: list[np.ndarray], maps: list[np.ndarray], n: int = 60) -> float | None:
    """Pixel threshold maximising Dice on (validation) anomaly masks."""
    gt = np.concatenate([m.reshape(-1) for m in masks]).astype(bool)
    if not gt.any():
        return None
    sc = np.concatenate([a.reshape(-1) for a in maps]).astype(np.float32)
    qs = np.unique(np.quantile(sc, np.linspace(0.80, 0.9995, n)))
    best_t, best_d = float(qs[-1]), -1.0
    gsum = int(gt.sum())
    for t in qs:
        p = sc >= t
        inter = int((p & gt).sum())
        d = 2 * inter / (int(p.sum()) + gsum)
        if d > best_d:
            best_d, best_t = d, float(t)
    return best_t


# ---- calibration ---------------------------------------------------------------------------

def platt_fit(y: np.ndarray, scores: np.ndarray) -> dict:
    """Fit P(defect | score) = sigmoid(a * score + b) on validation data (Platt scaling)."""
    from sklearn.linear_model import LogisticRegression

    s = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
    mu, sd = float(s.mean()), float(s.std() + 1e-12)
    lr = LogisticRegression(C=1.0).fit((s - mu) / sd, np.asarray(y).astype(int))
    a = float(lr.coef_[0, 0] / sd)
    b = float(lr.intercept_[0] - lr.coef_[0, 0] * mu / sd)
    return {"method": "platt", "a": a, "b": b}


def platt_apply(cal: dict, scores: np.ndarray | float) -> np.ndarray:
    z = cal["a"] * np.asarray(scores, dtype=np.float64) + cal["b"]
    return 1 / (1 + np.exp(-np.clip(z, -50, 50)))


def calibration_metrics(y: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(prob, dtype=np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    table = []
    for i in range(n_bins):
        sel = (p >= bins[i]) & (p < bins[i + 1] if i < n_bins - 1 else p <= bins[i + 1])
        if sel.any():
            conf, acc = float(p[sel].mean()), float(y[sel].mean())
            ece += sel.mean() * abs(conf - acc)
            table.append({"bin": i, "count": int(sel.sum()), "mean_prob": round(conf, 4), "frac_pos": round(acc, 4)})
    return {"ece": float(ece), "brier": float(np.mean((p - y) ** 2)), "reliability": table}
