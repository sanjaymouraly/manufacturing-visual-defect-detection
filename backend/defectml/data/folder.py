"""Import user-provided (custom) manufacturing datasets from a folder.

Supported layouts
-----------------
1. MVTec-style::

       train/good/*.png
       test/good/*.png
       test/<defect_type>/*.png
       ground_truth/<defect_type>/<stem>_mask.png   (optional)

2. Simple::

       normal/*.png                     (or good/, ok/)
       anomaly/*.png                    (or defective/, bad/, ng/)
       anomaly/<defect_type>/*.png      (optional sub-folders = user-provided defect classes)
       masks/<stem>.png                 (optional, non-zero = defect)
       groups.csv                       (optional: filename,group  e.g. production batch/session)

Defect types are recorded only from explicit user-provided folder names - never inferred.
Images are copied (re-encoded as PNG after safe decoding, optionally resized).
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from defectml.data.manifest import LABEL_ANOMALY, LABEL_NORMAL, DatasetInfo, Sample, save_dataset
from defectml.data.splits import SplitConfig, make_splits
from defectml.imaging import ALLOWED_EXTENSIONS, ImageDecodeError, load_image, resize_max_side

NORMAL_DIRS = {"normal", "good", "ok"}
ANOMALY_DIRS = {"anomaly", "defective", "bad", "ng", "defect"}


class DatasetImportError(ValueError):
    pass


def _images(d: Path) -> list[Path]:
    return sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS)


def _find_mask(mask_dirs: list[Path], stem: str) -> Path | None:
    for md in mask_dirs:
        for cand in (f"{stem}.png", f"{stem}_mask.png", f"{stem}.bmp", f"{stem}_mask.bmp"):
            if (md / cand).exists():
                return md / cand
    return None


def import_folder(src: Path, out_root: Path, name: str, product: str, license_text: str = "user-provided",
                  max_side: int = 512, seed: int = 42, source: str = "custom upload") -> tuple[Path, list[str]]:
    src = Path(src)
    if not src.is_dir():
        raise DatasetImportError("Source folder does not exist")
    entries: list[tuple[Path, str, str, Path | None, str]] = []  # path,label,type,mask,split
    if (src / "train").is_dir() and (src / "test").is_dir():  # MVTec-style
        for p in _images(src / "train" / "good") if (src / "train" / "good").is_dir() else []:
            entries.append((p, LABEL_NORMAL, "", None, "train"))
        for sub in sorted(x for x in (src / "test").iterdir() if x.is_dir()):
            for p in _images(sub):
                if sub.name == "good":
                    entries.append((p, LABEL_NORMAL, "", None, ""))
                else:
                    gt = src / "ground_truth" / sub.name
                    entries.append((p, LABEL_ANOMALY, sub.name, _find_mask([gt], p.stem), ""))
    else:
        children = {c.name.lower(): c for c in src.iterdir() if c.is_dir()}
        ndirs = [children[k] for k in children if k in NORMAL_DIRS]
        adirs = [children[k] for k in children if k in ANOMALY_DIRS]
        if not ndirs:
            raise DatasetImportError("No normal/ (or good/, ok/) folder found")
        mask_dirs = [children[k] for k in children if k in ("masks", "ground_truth")]
        for d in ndirs:
            for p in _images(d):
                entries.append((p, LABEL_NORMAL, "", None, ""))
        for d in adirs:
            for p in _images(d):
                rel_parent = p.parent.relative_to(d)
                dtype = rel_parent.parts[0] if rel_parent.parts else ""
                entries.append((p, LABEL_ANOMALY, dtype, _find_mask(mask_dirs, p.stem), ""))
    if not entries:
        raise DatasetImportError("No supported images found")
    groups: dict[str, str] = {}
    gfile = src / "groups.csv"
    if gfile.exists():
        with open(gfile, newline="", encoding="utf-8") as fh:
            for row in csv.reader(fh):
                if len(row) >= 2 and row[0] != "filename":
                    groups[row[0].strip()] = row[1].strip()
    out = out_root / name
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    samples: list[Sample] = []
    skipped: list[str] = []
    for idx, (p, label, dtype, mask_p, split) in enumerate(entries):
        try:
            img = resize_max_side(load_image(p), max_side)
        except ImageDecodeError as exc:
            skipped.append(f"{p.name}: {exc}")
            continue
        stem = f"{idx:05d}_{p.stem}"[:80]
        cv2.imwrite(str(out / "images" / f"{stem}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        mask_rel = ""
        if mask_p is not None:
            m = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                m = cv2.resize(((m > 0) * 255).astype(np.uint8), (img.shape[1], img.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
                mask_rel = f"masks/{stem}.png"
                cv2.imwrite(str(out / mask_rel), m)
        samples.append(Sample(path=f"images/{stem}.png", label=label, defect_type=dtype, mask=mask_rel,
                              group=groups.get(p.name, ""), split=split, source_id=str(p.relative_to(src))))
    if not any(s.label == LABEL_NORMAL for s in samples):
        raise DatasetImportError("Dataset must contain normal images to train anomaly detectors")
    make_splits(samples, SplitConfig(seed=seed))
    info = DatasetInfo(
        name=name, product=product, source=source, license=license_text, synthetic=False,
        defect_types_annotated=any(s.defect_type for s in samples),
        masks_available=any(s.mask for s in samples),
        preprocessing={"max_side": max_side}, split_policy={"protocol": "one_class", "seed": seed},
        created_at=datetime.now(UTC).isoformat(), redistributable=False,
        description="Custom dataset imported by a user.",
    )
    save_dataset(out, info, samples)
    return out, skipped
