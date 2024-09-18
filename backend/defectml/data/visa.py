"""VisA (Visual Anomaly) dataset acquisition and preparation.

Source:   https://github.com/amazon-science/spot-diff
Archive:  https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar
License:  CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/) - attribution required.
Citation: Zou et al., "SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection
          and Segmentation", ECCV 2022.

The dataset is downloaded by the user from the original host; DefectLens never redistributes it.
Defect-type strings come from the official ``image_anno.csv`` (genuine annotations).
"""

from __future__ import annotations

import csv
import io
import tarfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from defectml.data.manifest import LABEL_ANOMALY, LABEL_NORMAL, DatasetInfo, Sample, save_dataset
from defectml.data.splits import SplitConfig, make_splits
from defectml.imaging import load_image, resize_max_side, sha256_file

VISA_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
VISA_SPLIT_URL = "https://raw.githubusercontent.com/amazon-science/spot-diff/main/split_csv/1cls.csv"
# SHA-256 of VisA_20220922.tar as downloaded on 2026-10-17 (used to detect corrupt/changed archives).
VISA_SHA256 = "2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362"
VISA_LICENSE = "CC-BY-4.0"
VISA_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
VISA_CITATION = (
    "Zou, Y., Jeong, J., Pemula, L., Zhang, D., Dabeer, O. SPot-the-Difference Self-Supervised "
    "Pre-training for Anomaly Detection and Segmentation. ECCV 2022. arXiv:2207.14315"
)
VISA_CATEGORIES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2",
    "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]


class LicenseNotAccepted(RuntimeError):
    pass


def download_visa(dest_dir: Path, accept_license: bool, verify: bool = True) -> Path:
    if not accept_license:
        raise LicenseNotAccepted(
            "VisA is licensed CC BY 4.0. Re-run with --accept-license to confirm you will provide attribution."
        )
    dest_dir.mkdir(parents=True, exist_ok=True)
    tar_path = dest_dir / "VisA_20220922.tar"
    if not tar_path.exists():
        tmp = tar_path.with_suffix(".part")
        urllib.request.urlretrieve(VISA_URL, tmp)
        tmp.rename(tar_path)
    split_path = dest_dir / "1cls.csv"
    if not split_path.exists():
        urllib.request.urlretrieve(VISA_SPLIT_URL, split_path)
    if verify:
        digest = sha256_file(tar_path)
        if digest != VISA_SHA256:
            raise RuntimeError(f"VisA archive checksum mismatch: {digest}")
    return tar_path


def extract_categories(tar_path: Path, out_dir: Path, categories: list[str]) -> Path:
    """Safely extract selected categories (path traversal protected)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir.resolve()
    with tarfile.open(tar_path) as tf:
        members = []
        for m in tf.getmembers():
            top = m.name.split("/")[0]
            if top not in categories and top != "LICENSE-DATASET":
                continue
            target = (out_dir / m.name).resolve()
            if base not in target.parents and target != base:
                raise RuntimeError(f"Unsafe path in archive: {m.name}")
            if m.issym() or m.islnk():
                continue
            members.append(m)
        tf.extractall(out_dir, members=members)  # noqa: S202 - members validated above
    return out_dir


def _read_split_csv(path: Path, category: str) -> dict[str, str]:
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["object"] == category:
                out[row["image"].strip()] = row["split"].strip()
    return out


def _norm_label(text: str) -> str:
    parts = [p.strip().replace(" ", "_") for p in text.replace(";", ",").split(",") if p.strip()]
    return "|".join(sorted(set(parts)))


def prepare_visa_category(raw_root: Path, split_csv: Path, category: str, out_root: Path,
                          max_side: int = 512, seed: int = 42, archive_sha256: str = "") -> Path:
    """Convert one VisA category into the DefectLens dataset format.

    * images are resized so the longest side is ``max_side`` (masks with nearest-neighbour)
    * official 1-class train split is kept; the official test split is divided into
      ``val`` (threshold selection) and ``test`` (held-out reporting), stratified by label
    """
    if category not in VISA_CATEGORIES:
        raise ValueError(f"Unknown VisA category {category}")
    cat_dir = raw_root / category
    anno = cat_dir / "image_anno.csv"
    if not anno.exists():
        raise FileNotFoundError(f"{anno} not found - extract the category first")
    official = _read_split_csv(split_csv, category)
    out_dir = out_root / f"visa-{category}"
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    samples: list[Sample] = []
    with open(anno, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        rel = row["image"].strip()
        label_text = row["label"].strip()
        is_anom = label_text.lower() != "normal"
        img = load_image(raw_root / rel)
        small = resize_max_side(img, max_side)
        stem = ("A_" if is_anom else "N_") + Path(rel).stem
        img_rel = f"images/{stem}.png"
        cv2.imwrite(str(out_dir / img_rel), cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
        mask_rel = ""
        if is_anom and row.get("mask"):
            m = cv2.imread(str(raw_root / row["mask"].strip()), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                mb: np.ndarray = ((m > 0) * 255).astype(np.uint8)
                mb = cv2.resize(mb, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_rel = f"masks/{stem}.png"
                cv2.imwrite(str(out_dir / mask_rel), mb)
        split = official.get(rel, "")
        samples.append(Sample(
            path=img_rel,
            label=LABEL_ANOMALY if is_anom else LABEL_NORMAL,
            defect_type=_norm_label(label_text) if is_anom else "",
            mask=mask_rel,
            group="",  # VisA provides no batch/session/camera ids
            split="train" if split == "train" else "",
            source_id=rel,
        ))
    cfg = SplitConfig(seed=seed)
    make_splits(samples, cfg)
    info = DatasetInfo(
        name=f"visa-{category}",
        product=f"visa-{category}",
        source="VisA (Amazon Science, spot-diff)",
        source_url="https://github.com/amazon-science/spot-diff",
        license=VISA_LICENSE,
        license_url=VISA_LICENSE_URL,
        citation=VISA_CITATION,
        version="VisA_20220922",
        description=f"VisA '{category}' prepared for one-class anomaly detection.",
        synthetic=False,
        defect_types_annotated=True,
        masks_available=True,
        preprocessing={"max_side": max_side, "mask_binarization": "value>0", "format": "png"},
        split_policy={"protocol": "one_class", "official_split": "1cls.csv", "seed": seed,
                      "val_fraction_of_holdout": cfg.val_fraction_of_holdout,
                      "val_fraction_anomaly": cfg.val_fraction_anomaly},
        archive_sha256=archive_sha256,
        created_at=datetime.now(UTC).isoformat(),
        redistributable=False,  # we do not redistribute; users download from the original host
    )
    lic = raw_root / "LICENSE-DATASET"
    if lic.exists():
        (out_dir / "LICENSE-DATASET.txt").write_bytes(lic.read_bytes())
    (out_dir / "ATTRIBUTION.txt").write_text(
        io.StringIO(f"{info.source}\n{info.source_url}\nLicense: {info.license} {info.license_url}\n"
                    f"Citation: {info.citation}\nModified: resized to max side {max_side}px; masks binarized.\n"
                    ).getvalue(), encoding="utf-8")
    save_dataset(out_dir, info, samples)
    return out_dir
