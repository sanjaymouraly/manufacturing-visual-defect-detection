"""Turn anomaly maps into defect regions (boxes, polygons, areas) and visual overlays."""

from __future__ import annotations

import cv2
import numpy as np


def map_to_image(anomaly_map: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    return cv2.resize(anomaly_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def extract_regions(anomaly_map: np.ndarray, threshold: float, image_shape: tuple[int, int],
                    min_area_fraction: float = 0.0005, max_regions: int = 20) -> tuple[list[dict], np.ndarray]:
    """Connected regions where the (resized) anomaly map exceeds ``threshold``.

    Returns region dicts in ORIGINAL image pixel coordinates and the binary mask (uint8 0/1).
    """
    m = map_to_image(anomaly_map, image_shape)
    binary: np.ndarray = (m >= threshold).astype(np.uint8)
    if binary.any():
        k = max(1, round(min(image_shape) / 150))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    total = image_shape[0] * image_shape[1]
    regions = []
    keep = np.zeros_like(binary)
    for r in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[r])
        if area < min_area_fraction * total:
            continue
        comp = (lab == r).astype(np.uint8)
        keep |= comp
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        poly: list[list[int]] = []
        if contours:
            c = max(contours, key=cv2.contourArea)
            eps = 0.01 * cv2.arcLength(c, True)
            poly = [[int(p[0][0]), int(p[0][1])] for p in cv2.approxPolyDP(c, eps, True)]
        vals = m[lab == r]
        regions.append({
            "bbox": [x, y, w, h],
            "polygon": poly,
            "area_px": area,
            "area_fraction": round(area / total, 6),
            "max_score": round(float(vals.max()), 5),
            "mean_score": round(float(vals.mean()), 5),
        })
    regions.sort(key=lambda r: -float(r["max_score"]))  # type: ignore[arg-type]
    return regions[:max_regions], keep


def normalise_map(anomaly_map: np.ndarray, low: float, high: float) -> np.ndarray:
    if high <= low:
        high = low + 1e-6
    return np.clip((anomaly_map - low) / (high - low), 0, 1)


def render_heatmap(anomaly_map: np.ndarray, shape: tuple[int, int], low: float, high: float) -> np.ndarray:
    """RGB heatmap (JET) on a fixed, model-specific scale so colours are comparable across images."""
    m = normalise_map(map_to_image(anomaly_map, shape), low, high)
    colored = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def render_overlay(image: np.ndarray, heatmap: np.ndarray, mask: np.ndarray | None, alpha: float = 0.45,
                   regions: list[dict] | None = None) -> np.ndarray:
    out = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    if mask is not None and mask.any():
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (255, 255, 255), max(1, min(image.shape[:2]) // 300))
    for r in regions or []:
        x, y, w, h = r["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 40, 40), max(1, min(image.shape[:2]) // 250))
    return out
