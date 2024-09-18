"""Dataset validation and quality reporting.

Checks: readable images, formats, dimensions, mask presence/alignment, exact and near
duplicates, class balance, defect-type distribution, split leakage.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from defectml.data.manifest import Dataset
from defectml.data.splits import hamming_pairs, leakage_report, split_summary
from defectml.imaging import ImageDecodeError, dhash, load_image, sha256_file


def validate_dataset(ds: Dataset, near_dup_distance: int = 3, write: bool = True) -> dict:
    errors: list[dict] = []
    warnings: list[str] = []
    dims: Counter = Counter()
    formats: Counter = Counter()
    shas: list[str] = []
    hashes: list[np.ndarray] = []
    ok_idx: list[int] = []
    mask_issues: list[dict] = []
    for i, s in enumerate(ds.samples):
        p = ds.abspath(s.path)
        try:
            img = load_image(p)
        except ImageDecodeError as exc:
            errors.append({"path": s.path, "error": str(exc)})
            continue
        ok_idx.append(i)
        dims[f"{img.shape[1]}x{img.shape[0]}"] += 1
        formats[p.suffix.lower()] += 1
        shas.append(sha256_file(p))
        hashes.append(dhash(img))
        if s.mask:
            mp = ds.abspath(s.mask)
            if not mp.exists():
                mask_issues.append({"path": s.path, "issue": "mask file missing"})
            else:
                m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                if m is None:
                    mask_issues.append({"path": s.path, "issue": "mask unreadable"})
                elif m.shape != img.shape[:2]:
                    mask_issues.append({"path": s.path, "issue": f"mask shape {m.shape} != image {img.shape[:2]}"})
                elif s.is_anomaly and not (m > 0).any():
                    mask_issues.append({"path": s.path, "issue": "anomalous sample has empty mask"})
                elif not s.is_anomaly and (m > 0).any():
                    mask_issues.append({"path": s.path, "issue": "normal sample has non-empty mask"})
        elif s.is_anomaly and ds.info.masks_available:
            mask_issues.append({"path": s.path, "issue": "anomalous sample without mask"})

    ok_samples = [ds.samples[i] for i in ok_idx]
    sha_counter = Counter(shas)
    exact_dups = [[ok_samples[i].path for i, h in enumerate(shas) if h == d] for d, c in sha_counter.items() if c > 1]
    harr = np.array(hashes) if hashes else np.zeros((0, 256), dtype=bool)
    near = hamming_pairs(harr, near_dup_distance) if len(harr) > 1 else []
    labels = Counter(s.label for s in ds.samples)
    n_norm, n_anom = labels.get("normal", 0), labels.get("anomaly", 0)
    imbalance = round(n_norm / n_anom, 2) if n_anom else None
    defect_types: Counter = Counter()
    if ds.info.defect_types_annotated:
        for s in ds.samples:
            if s.is_anomaly:
                for t in (s.defect_type or "unspecified").split("|"):
                    defect_types[t] += 1
    if n_anom == 0:
        warnings.append("No anomalous samples: thresholds will be set from normal-score percentiles only.")
    if imbalance and imbalance > 20:
        warnings.append(f"Strong class imbalance (normal:anomaly = {imbalance}:1).")
    if len(dims) > 1:
        warnings.append(f"{len(dims)} distinct image resolutions found; images are resized for modelling.")
    leak = leakage_report(ok_samples, shas, harr, near_dup_distance)
    if not leak["leakage_free"]:
        warnings.append("Split leakage detected (group overlap or exact duplicates across splits).")
    report = {
        "dataset": ds.info.name,
        "product": ds.info.product,
        "num_samples": len(ds.samples),
        "num_readable": len(ok_idx),
        "labels": dict(labels),
        "normal_to_anomaly_ratio": imbalance,
        "defect_types_annotated": ds.info.defect_types_annotated,
        "defect_types": dict(defect_types),
        "splits": split_summary(ds.samples),
        "dimensions": dict(dims.most_common(10)),
        "formats": dict(formats),
        "errors": errors[:200],
        "mask_issues": mask_issues[:200],
        "exact_duplicate_groups": exact_dups[:100],
        "near_duplicate_pairs": len(near),
        "near_duplicate_examples": [[ok_samples[i].path, ok_samples[j].path, d] for i, j, d in near[:20]],
        "leakage": {k: (v[:50] if isinstance(v, list) else v) for k, v in leak.items()},
        "warnings": warnings,
        "valid": not errors and not mask_issues and leak["leakage_free"],
    }
    if write:
        (ds.root / "quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        write_preview(ds, ds.root / "preview.jpg")
    return report


def write_preview(ds: Dataset, out: Path, per_row: int = 6, tile: int = 128) -> Path:
    """Grid preview: first row normal samples, second row anomalous samples with mask outline."""
    normals = [s for s in ds.samples if not s.is_anomaly][:per_row]
    anomalies = [s for s in ds.samples if s.is_anomaly][:per_row]
    rows = []
    for group in (normals, anomalies):
        tiles = []
        for s in group:
            try:
                img = load_image(ds.abspath(s.path))
            except ImageDecodeError:
                img = np.zeros((tile, tile, 3), np.uint8)
            img = cv2.resize(img, (tile, tile), interpolation=cv2.INTER_AREA)
            if s.mask:
                m = cv2.imread(str(ds.abspath(s.mask)), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    m = cv2.resize((m > 0).astype(np.uint8), (tile, tile), interpolation=cv2.INTER_NEAREST)
                    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img, contours, -1, (255, 0, 0), 1)
            tiles.append(img)
        while len(tiles) < per_row:
            tiles.append(np.full((tile, tile, 3), 40, np.uint8))
        rows.append(np.concatenate(tiles, axis=1))
    grid = np.concatenate(rows, axis=0)
    cv2.imwrite(str(out), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out
