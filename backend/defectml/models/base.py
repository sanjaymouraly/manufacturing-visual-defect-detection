"""Common model contract.

Every anomaly model maps a list of RGB uint8 images to :class:`Prediction` objects:

* ``score``       raw anomaly score (higher = more anomalous). NOT a probability.
* ``anomaly_map`` float32 (input_size x input_size) map on the same scale as ``score`` or None
* ``ood_score``   distance of the image's global embedding from the training distribution

Persisted artifacts: ``weights.npz`` (numpy arrays only, loaded with ``allow_pickle=False``) and
``model.json`` (algorithm, config, thresholds, calibration, metrics, provenance).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.covariance import LedoitWolf

ProgressFn = Callable[[float, str], None]
CancelFn = Callable[[], bool]


class TrainingCancelled(RuntimeError):
    pass


@dataclass
class Prediction:
    score: float
    anomaly_map: np.ndarray | None
    ood_score: float | None = None


class AnomalyModel(ABC):
    algorithm: str = "base"
    supports_localization: bool = True
    requires_torch: bool = True

    def __init__(self, **config):
        self.config = {**self.default_config(), **config}
        self.history: list[dict] = []

    @classmethod
    def default_config(cls) -> dict:
        return {"input_size": 224, "batch_size": 16, "device": "auto"}

    @abstractmethod
    def fit(self, images: Iterable[np.ndarray], n_images: int, progress: ProgressFn | None = None,
            cancel: CancelFn | None = None) -> None: ...

    @abstractmethod
    def predict_batch(self, images: list[np.ndarray]) -> list[Prediction]: ...

    @abstractmethod
    def state_arrays(self) -> dict[str, np.ndarray]: ...

    @abstractmethod
    def load_state(self, arrays: dict[str, np.ndarray]) -> None: ...

    def predict(self, images: list[np.ndarray]) -> list[Prediction]:
        out: list[Prediction] = []
        bs = int(self.config.get("batch_size", 16))
        for i in range(0, len(images), bs):
            out.extend(self.predict_batch(images[i:i + bs]))
        return out

    # ---- helpers -------------------------------------------------------------------------
    def _check_cancel(self, cancel: CancelFn | None) -> None:
        if cancel is not None and cancel():
            raise TrainingCancelled("Training cancelled by user")


def smooth_map(m: np.ndarray, sigma: float) -> np.ndarray:
    return gaussian_filter(m, sigma=sigma).astype(np.float32) if sigma > 0 else m.astype(np.float32)


class GaussianOOD:
    """Mahalanobis distance of global embeddings to the training distribution (Ledoit-Wolf)."""

    def __init__(self):
        self.mean: np.ndarray | None = None
        self.precision: np.ndarray | None = None

    def fit(self, emb: np.ndarray) -> None:
        lw = LedoitWolf().fit(emb.astype(np.float64))
        self.mean = lw.location_.astype(np.float32)
        self.precision = lw.precision_.astype(np.float32)

    def score(self, emb: np.ndarray) -> np.ndarray:
        if self.mean is None or self.precision is None:
            return np.zeros(len(emb), np.float32)
        d = emb - self.mean
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, self.precision, d), 0)).astype(np.float32)

    def arrays(self) -> dict[str, np.ndarray]:
        if self.mean is None or self.precision is None:
            return {}
        return {"ood_mean": self.mean, "ood_precision": self.precision}

    def load(self, arrays: dict[str, np.ndarray]) -> None:
        if "ood_mean" in arrays:
            self.mean = arrays["ood_mean"]
            self.precision = arrays["ood_precision"]


def save_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]  # numpy stub mis-types **kwargs


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def read_model_json(model_dir: Path) -> dict:
    return json.loads((Path(model_dir) / "model.json").read_text(encoding="utf-8"))
