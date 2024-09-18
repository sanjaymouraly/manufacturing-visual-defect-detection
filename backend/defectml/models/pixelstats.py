"""Lightweight classical baseline: per-pixel colour statistics ("golden template" z-score).

Learns the per-pixel mean and standard deviation (CIE Lab, blurred, 128x128) of aligned normal
images. The anomaly map is the maximum channel-wise z-score. No deep learning, no pretrained
weights, runs anywhere - but only meaningful for well-aligned products with fixed fixturing.
"""

from __future__ import annotations

import time
from collections.abc import Iterable

import cv2
import numpy as np

from defectml.imaging import resize_rgb
from defectml.models.base import AnomalyModel, CancelFn, Prediction, ProgressFn, smooth_map


class PixelStats(AnomalyModel):
    algorithm = "pixelstats"
    requires_torch = False

    @classmethod
    def default_config(cls) -> dict:
        return {**super().default_config(), "work_size": 128, "pre_blur": 1.0, "std_floor_pct": 10.0,
                "blur_sigma": 2.0, "top_fraction": 0.005, "batch_size": 32}

    def __init__(self, **config):
        super().__init__(**config)
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.global_mean: np.ndarray | None = None
        self.global_std: np.ndarray | None = None

    def _prep(self, im: np.ndarray) -> np.ndarray:
        s = self.config["work_size"]
        x = resize_rgb(im, s)
        if self.config["pre_blur"] > 0:
            x = cv2.GaussianBlur(x, (0, 0), self.config["pre_blur"])
        return cv2.cvtColor(x, cv2.COLOR_RGB2LAB).astype(np.float32)

    @staticmethod
    def _global(x: np.ndarray) -> np.ndarray:
        return np.concatenate([x.mean(axis=(0, 1)), x.std(axis=(0, 1))])

    def fit(self, images: Iterable[np.ndarray], n_images: int, progress: ProgressFn | None = None,
            cancel: CancelFn | None = None) -> None:
        t0 = time.time()
        s1: np.ndarray | None = None
        s2: np.ndarray | None = None
        n = 0
        glob = []
        for im in images:
            x = self._prep(im)
            if s1 is None:
                s1 = np.zeros_like(x, dtype=np.float64)
                s2 = np.zeros_like(x, dtype=np.float64)
            assert s1 is not None and s2 is not None
            s1 += x
            s2 += x.astype(np.float64) ** 2
            glob.append(self._global(x))
            n += 1
            if n % 32 == 0:
                self._check_cancel(cancel)
                if progress:
                    progress(n / max(n_images, 1), f"Accumulated {n}/{n_images} images")
        if n < 2 or s1 is None or s2 is None:
            raise ValueError("PixelStats needs at least 2 training images")
        mean = s1 / n
        var = np.maximum(s2 / n - mean ** 2, 0) * n / (n - 1)
        std = np.sqrt(var)
        floor = np.percentile(std, self.config["std_floor_pct"], axis=(0, 1), keepdims=True)
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std, np.maximum(floor, 1.0)).astype(np.float32)
        g = np.array(glob, np.float32)
        self.global_mean, self.global_std = g.mean(0), g.std(0) + 1e-3
        self.history.append({"step": "pixel_statistics", "images": n, "seconds": round(time.time() - t0, 2)})
        if progress:
            progress(1.0, "PixelStats fitted")

    def predict_batch(self, images: list[np.ndarray]) -> list[Prediction]:
        if self.mean is None or self.std is None:
            raise RuntimeError("PixelStats model is not fitted")
        out = []
        size = self.config["input_size"]
        for im in images:
            x = self._prep(im)
            z = np.abs(x - self.mean) / self.std
            m = smooth_map(z.max(axis=-1), self.config["blur_sigma"])
            m = cv2.resize(m, (size, size), interpolation=cv2.INTER_LINEAR)
            k = max(1, int(m.size * self.config["top_fraction"]))
            score = float(np.sort(m.ravel())[-k:].mean())
            ood = float(np.abs((self._global(x) - self.global_mean) / self.global_std).max())
            out.append(Prediction(score=score, anomaly_map=m.astype(np.float32), ood_score=ood))
        return out

    def state_arrays(self) -> dict[str, np.ndarray]:
        assert self.mean is not None and self.std is not None
        assert self.global_mean is not None and self.global_std is not None
        return {"mean": self.mean, "std": self.std, "global_mean": self.global_mean, "global_std": self.global_std}

    def load_state(self, arrays: dict[str, np.ndarray]) -> None:
        self.mean, self.std = arrays["mean"], arrays["std"]
        self.global_mean, self.global_std = arrays["global_mean"], arrays["global_std"]
