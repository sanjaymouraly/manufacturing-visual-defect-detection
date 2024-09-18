"""PaDiM (Defard et al., "PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection
and Localization", ICPR 2021).

Resource-aware variant: features from ResNet18 layer1-3 are resampled to a 28x28 grid
(layer1 average-pooled, layer3 upsampled), ``n_features`` random channels are kept, and a
multivariate Gaussian (mean + regularised covariance) is estimated per grid position from
streaming sums (constant memory in the number of training images).
"""

from __future__ import annotations

import time
from collections.abc import Iterable

import numpy as np

from defectml.models.backbone import get_extractor
from defectml.models.base import AnomalyModel, CancelFn, GaussianOOD, Prediction, ProgressFn, smooth_map


class PaDiM(AnomalyModel):
    algorithm = "padim"

    @classmethod
    def default_config(cls) -> dict:
        return {**super().default_config(), "backbone": "resnet18", "n_features": 100, "grid": 28,
                "cov_reg": 0.01, "blur_sigma": 4.0, "seed": 0}

    def __init__(self, **config):
        super().__init__(**config)
        self.mean: np.ndarray | None = None
        self.inv_cov: np.ndarray | None = None
        self.channel_idx: np.ndarray | None = None
        self.ood = GaussianOOD()

    def _extractor(self):
        return get_extractor(self.config["backbone"], self.config["device"], self.config["input_size"])

    def _embed(self, images: list[np.ndarray]):
        import torch
        import torch.nn.functional as F

        ext = self._extractor()
        feats = ext(images)
        g = self.config["grid"]
        parts = [F.adaptive_avg_pool2d(feats["layer1"], g), F.adaptive_avg_pool2d(feats["layer2"], g)
                 if feats["layer2"].shape[-1] != g else feats["layer2"],
                 F.interpolate(feats["layer3"], size=(g, g), mode="bilinear", align_corners=False)]
        f = torch.cat(parts, dim=1)
        if self.channel_idx is None:
            rng = np.random.default_rng(self.config["seed"])
            k = min(self.config["n_features"], f.shape[1])
            self.channel_idx = np.sort(rng.choice(f.shape[1], k, replace=False)).astype(np.int64)
        f = f[:, torch.from_numpy(self.channel_idx).to(f.device)]
        b, c, h, w = f.shape
        return f.permute(0, 2, 3, 1).reshape(b, h * w, c), ext.global_embedding(feats)

    def fit(self, images: Iterable[np.ndarray], n_images: int, progress: ProgressFn | None = None,
            cancel: CancelFn | None = None) -> None:
        s1 = s2 = None
        n = 0
        embs = []
        batch: list[np.ndarray] = []
        t0 = time.time()

        def flush():
            nonlocal s1, s2, n
            f, emb = self._embed(batch)
            x = f.cpu().numpy().astype(np.float64)  # (B, P, C)
            if s1 is None:
                s1 = np.zeros(x.shape[1:], np.float64)
                s2 = np.zeros((x.shape[1], x.shape[2], x.shape[2]), np.float64)
            s1 += x.sum(0)
            s2 += np.einsum("bpi,bpj->pij", x, x)
            n += len(batch)
            embs.append(emb)
            batch.clear()
            if progress:
                progress(0.9 * n / max(n_images, 1), f"Accumulated statistics for {n}/{n_images} images")

        for im in images:
            batch.append(im)
            if len(batch) >= self.config["batch_size"]:
                self._check_cancel(cancel)
                flush()
        if batch:
            flush()
        if n < 2 or s1 is None or s2 is None:
            raise ValueError("PaDiM needs at least 2 training images")
        mean = s1 / n
        cov = (s2 - n * np.einsum("pi,pj->pij", mean, mean)) / (n - 1)
        cov += self.config["cov_reg"] * np.eye(cov.shape[-1])[None]
        self.mean = mean.astype(np.float32)
        self.inv_cov = np.linalg.inv(cov).astype(np.float32)
        self.ood.fit(np.concatenate(embs))
        self.history.append({"step": "gaussian_fit", "images": n, "positions": int(mean.shape[0]),
                             "features": int(mean.shape[1]), "seconds": round(time.time() - t0, 2)})
        if progress:
            progress(1.0, "PaDiM fitted")

    def predict_batch(self, images: list[np.ndarray]) -> list[Prediction]:
        import torch
        import torch.nn.functional as F

        if self.mean is None or self.inv_cov is None:
            raise RuntimeError("PaDiM model is not fitted")
        f, emb = self._embed(images)
        with torch.inference_mode():
            d = f - torch.from_numpy(self.mean).to(f.device)
            m = torch.einsum("bpi,pij,bpj->bp", d, torch.from_numpy(self.inv_cov).to(f.device), d)
            m = torch.sqrt(torch.clamp(m, min=0))
            g = self.config["grid"]
            size = self.config["input_size"]
            up = F.interpolate(m.reshape(-1, 1, g, g), size=(size, size), mode="bilinear", align_corners=False)
        maps = up[:, 0].cpu().numpy()
        ood = self.ood.score(emb)
        out = []
        for i in range(len(images)):
            sm = smooth_map(maps[i], self.config["blur_sigma"])
            out.append(Prediction(score=float(sm.max()), anomaly_map=sm, ood_score=float(ood[i])))
        return out

    def state_arrays(self) -> dict[str, np.ndarray]:
        assert self.mean is not None and self.inv_cov is not None and self.channel_idx is not None
        return {"mean": self.mean, "inv_cov": self.inv_cov, "channel_idx": self.channel_idx,
                **self.ood.arrays()}

    def load_state(self, arrays: dict[str, np.ndarray]) -> None:
        self.mean = arrays["mean"].astype(np.float32)
        self.inv_cov = arrays["inv_cov"].astype(np.float32)
        self.channel_idx = arrays["channel_idx"].astype(np.int64)
        self.ood.load(arrays)
