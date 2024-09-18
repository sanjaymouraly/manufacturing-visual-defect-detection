"""Dataset manifest format.

A prepared dataset is a directory containing::

    dataset.json      metadata: name, product, provenance, license, citation, preprocessing, version
    manifest.csv      one row per image (see :class:`Sample`)
    images/...        image files (paths in manifest are relative to the dataset directory)
    masks/...         optional binary masks (non-zero = defect)

``defect_type`` is only populated when the source provides genuine per-image defect annotations
(``dataset.json -> defect_types_annotated``). It is never derived from binary anomaly masks.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

LABEL_NORMAL = "normal"
LABEL_ANOMALY = "anomaly"
SPLITS = ("train", "val", "test")


@dataclass
class Sample:
    path: str
    label: str  # "normal" | "anomaly"
    defect_type: str = ""  # genuine annotation only; multi-label joined with "|"
    mask: str = ""
    group: str = ""  # production batch / session / camera id used for leakage-free splitting
    split: str = ""
    source_id: str = ""  # identifier in the original dataset

    @property
    def is_anomaly(self) -> bool:
        return self.label == LABEL_ANOMALY

    @property
    def y(self) -> int:
        return 1 if self.is_anomaly else 0


@dataclass
class DatasetInfo:
    name: str
    product: str
    source: str
    license: str
    license_url: str = ""
    citation: str = ""
    source_url: str = ""
    version: str = "1"
    description: str = ""
    synthetic: bool = False
    defect_types_annotated: bool = False
    masks_available: bool = False
    preprocessing: dict = field(default_factory=dict)
    split_policy: dict = field(default_factory=dict)
    archive_sha256: str = ""
    manifest_sha256: str = ""
    created_at: str = ""
    redistributable: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


FIELDS = [f for f in Sample.__dataclass_fields__]


def write_manifest(path: Path, samples: list[Sample]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for s in samples:
            w.writerow(asdict(s))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_manifest(path: Path) -> list[Sample]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        out.append(Sample(**{k: (r.get(k) or "") for k in FIELDS}))
    return out


class Dataset:
    """A prepared dataset on disk."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        meta_path = self.root / "dataset.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No dataset.json in {self.root}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        known = DatasetInfo.__dataclass_fields__
        self.info = DatasetInfo(**{k: v for k, v in meta.items() if k in known})
        self.samples = read_manifest(self.root / "manifest.csv")

    def abspath(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if self.root.resolve() not in p.parents:
            raise ValueError(f"Manifest path escapes dataset root: {rel}")
        return p

    def split(self, name: str) -> list[Sample]:
        return [s for s in self.samples if s.split == name]

    def save(self) -> None:
        self.info.manifest_sha256 = write_manifest(self.root / "manifest.csv", self.samples)
        (self.root / "dataset.json").write_text(json.dumps(self.info.to_dict(), indent=2), encoding="utf-8")


def save_dataset(root: Path, info: DatasetInfo, samples: list[Sample]) -> Dataset:
    root.mkdir(parents=True, exist_ok=True)
    info.manifest_sha256 = write_manifest(root / "manifest.csv", samples)
    (root / "dataset.json").write_text(json.dumps(info.to_dict(), indent=2), encoding="utf-8")
    return Dataset(root)
