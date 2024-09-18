"""Reproducible, group-aware, leakage-checked dataset splitting.

Anomaly-detection protocol (default, ``one_class``):
    * ``train`` contains NORMAL images only (models learn normality)
    * ``val`` contains normal + anomalous images and is used ONLY for threshold selection,
      calibration and early stopping
    * ``test`` is held out and used ONLY for final reporting

Samples sharing a ``group`` (production batch, capture session, camera, physical part, or a
near-duplicate cluster) are always assigned to the same split to prevent leakage.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from defectml.data.manifest import Sample


@dataclass
class SplitConfig:
    seed: int = 42
    val_fraction_normal: float = 0.1  # of all normals (when no official split exists)
    test_fraction_normal: float = 0.15
    val_fraction_of_holdout: float = 0.35  # when re-splitting an official test split into val/test
    val_fraction_anomaly: float = 0.35  # of anomalies (rest go to test)
    keep_official_train: bool = True


def _order_key(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def _group_key(s: Sample) -> str:
    return s.group or f"__single__:{s.path}"


def _assign(groups: dict[str, list[Sample]], fractions: list[tuple[str, float]], seed: int) -> None:
    """Assign whole groups to splits so that each split receives ~fraction of samples."""
    keys = sorted(groups, key=lambda k: _order_key(seed, k))
    total = sum(len(groups[k]) for k in keys)
    targets = []
    acc = 0.0
    for name, frac in fractions:
        acc += frac
        targets.append((name, acc * total))
    count = 0
    for k in keys:
        n = len(groups[k])
        # place group in the first split whose cumulative target is not yet reached
        name = fractions[-1][0]
        for split_name, cum in targets:
            if count + n / 2 <= cum:
                name = split_name
                break
        for s in groups[k]:
            s.split = name
        count += n


def make_splits(samples: list[Sample], cfg: SplitConfig | None = None) -> list[Sample]:
    cfg = cfg or SplitConfig()
    has_official_train = cfg.keep_official_train and any(s.split == "train" for s in samples)

    normals = [s for s in samples if not s.is_anomaly]
    anomalies = [s for s in samples if s.is_anomaly]

    # anomalies never go to train in the one-class protocol
    for s in anomalies:
        if s.split == "train":
            s.split = ""

    if has_official_train:
        holdout_normals = [s for s in normals if s.split != "train"]
        n_groups: dict[str, list[Sample]] = defaultdict(list)
        for s in holdout_normals:
            n_groups[_group_key(s)].append(s)
        _assign(n_groups, [("val", cfg.val_fraction_of_holdout), ("test", 1 - cfg.val_fraction_of_holdout)],
                cfg.seed)
    else:
        n_groups = defaultdict(list)
        for s in normals:
            n_groups[_group_key(s)].append(s)
        tr = 1 - cfg.val_fraction_normal - cfg.test_fraction_normal
        _assign(n_groups, [("train", tr), ("val", cfg.val_fraction_normal), ("test", cfg.test_fraction_normal)],
                cfg.seed)

    a_groups: dict[str, list[Sample]] = defaultdict(list)
    for s in anomalies:
        a_groups[_group_key(s)].append(s)
    _assign(a_groups, [("val", cfg.val_fraction_anomaly), ("test", 1 - cfg.val_fraction_anomaly)], cfg.seed + 1)

    # Groups must not straddle splits: if a group got split between normal/anomaly assignment,
    # move the whole group to the split of its anomalies (held-out side).
    by_group: dict[str, set[str]] = defaultdict(set)
    for s in samples:
        if s.group:
            by_group[s.group].add(s.split)
    for g, splits in by_group.items():
        if len(splits) > 1:
            target = "test" if "test" in splits else ("val" if "val" in splits else "train")
            for s in samples:
                if s.group == g:
                    s.split = target
    return samples


def split_summary(samples: list[Sample]) -> dict:
    out: dict[str, dict[str, int]] = {}
    for s in samples:
        d = out.setdefault(s.split or "unassigned", {"normal": 0, "anomaly": 0})
        d[s.label] = d.get(s.label, 0) + 1
    return out


def hamming_pairs(hashes: np.ndarray, max_distance: int, chunk: int = 128) -> list[tuple[int, int, int]]:
    """Return (i, j, distance) for all pairs i<j with Hamming distance <= max_distance."""
    hashes = hashes.astype(np.uint8)
    n = len(hashes)
    pairs = []
    for start in range(0, n, chunk):
        block = hashes[start:start + chunk]
        d = (block[:, None, :] != hashes[None, :, :]).sum(-1)
        ii, jj = np.nonzero(d <= max_distance)
        for i, j in zip(ii, jj, strict=True):
            gi = start + int(i)
            if gi < j:
                pairs.append((gi, int(j), int(d[i, j])))
    return pairs


def leakage_report(samples: list[Sample], sha: list[str] | None = None, dhashes: np.ndarray | None = None,
                   near_dup_distance: int = 3) -> dict:
    """Check for train/val/test leakage: group overlap, exact duplicates and near-duplicates."""
    report: dict = {"group_overlap": [], "exact_duplicates_across_splits": [], "near_duplicates_across_splits": []}
    groups: dict[str, set[str]] = defaultdict(set)
    for s in samples:
        if s.group:
            groups[s.group].add(s.split)
    report["group_overlap"] = sorted(g for g, sp in groups.items() if len(sp) > 1)
    if sha is not None:
        seen: dict[str, int] = {}
        for i, h in enumerate(sha):
            if h in seen and samples[seen[h]].split != samples[i].split:
                report["exact_duplicates_across_splits"].append([samples[seen[h]].path, samples[i].path])
            seen.setdefault(h, i)
    if dhashes is not None and len(dhashes) > 1:
        for i, j, d in hamming_pairs(dhashes, near_dup_distance):
            if samples[i].split != samples[j].split:
                report["near_duplicates_across_splits"].append([samples[i].path, samples[j].path, d])
    report["leakage_free"] = not (report["group_overlap"] or report["exact_duplicates_across_splits"])
    report["near_duplicate_count"] = len(report["near_duplicates_across_splits"])
    return report
