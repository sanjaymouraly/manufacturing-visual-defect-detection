"""PatchCore (Roth et al., "Towards Total Recall in Industrial Anomaly Detection", CVPR 2022).

Implementation notes (resource-aware variant for CPU / small-RAM machines):

* mid-level features from ResNet layer2 + layer3 (layer3 upsampled), 3x3 local average pooling
* per-image random patch subsampling before coreset selection to bound RAM
  (``patches_per_image``), then greedy k-center coreset selection on a random projection
* anomaly score of a patch = distance to nearest memory-bank patch; image score = max of the
  Gaussian-smoothed anomaly map (sigma = 4 px at 224 px input)
"""

from __future__ import annotations

import time
from collections.abc import Iterable

import numpy as np

from defectml.models.backbone import get_extractor
from defectml.models.base import AnomalyModel, CancelFn, GaussianOOD, Prediction, ProgressFn, smooth_map


def greedy_coreset(features: np.ndarray, n_select: int, proj_dim: int = 128, seed: int = 0,
                   cancel: CancelFn | None = None) -> np.ndarray:
    """Greedy k-center selection (approximate, on a Johnson-Lindenstrauss projection)."""
    import torch

    n = len(features)
    if n_select >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    d_in = features.shape[1]
    proj = (torch.from_numpy(rng.normal(0, 1 / np.sqrt(proj_dim), (d_in, proj_dim)).astype(np.float32))
            if d_in > proj_dim else None)
    # Project in chunks so a float16 bank never has to be materialised as float32 in full.
    parts = []
    for a in range(0, n, 16384):
        c = torch.from_numpy(np.asarray(features[a:a + 16384], dtype=np.float32))
        parts.append(c @ proj if proj is not None else c)
    x = torch.cat(parts)
    del parts
    sq = (x * x).sum(1)
    selected = np.empty(n_select, dtype=np.int64)
    idx = int(rng.integers(0, n))
    min_d = torch.full((n,), float("inf"))
    for i in range(n_select):
        selected[i] = idx
        c = x[idx]
        # ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2  (no n x d temporary per iteration)
        d = sq - 2 * (x @ c) + sq[idx]
        torch.minimum(min_d, d, out=min_d)
        idx = int(torch.argmax(min_d))
        if cancel is not None and i % 500 == 0 and cancel():
            break
    return selected


class PatchCore(AnomalyModel):
    algorithm = "patchcore"

    @classmethod
    def default_config(cls) -> dict:
        return {
            **super().default_config(),
            "backbone": "resnet18",
            "patches_per_image": 160,
            "coreset_ratio": 0.1,
            "max_memory_bank": 16000,
            "projection_dim": 128,
            "knn_chunk": 4096,
            "blur_sigma": 4.0,
            "seed": 0,
        }

    def __init__(self, **config):
        super().__init__(**config)
        self.memory_bank: np.ndarray | None = None
        self.ood = GaussianOOD()

    def _extractor(self):
        return get_extractor(self.config["backbone"], self.config["device"], self.config["input_size"])

    def _patch_features(self, images: list[np.ndarray]):
        import torch
        import torch.nn.functional as F

        ext = self._extractor()
        feats = ext(images)
        l2, l3 = feats["layer2"], feats["layer3"]
        l3u = F.interpolate(l3, size=l2.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([l2, l3u], dim=1)
        f = F.avg_pool2d(f, 3, stride=1, padding=1)
        b, c, h, w = f.shape
        patches = f.permute(0, 2, 3, 1).reshape(b, h * w, c)
        return patches, (h, w), ext.global_embedding(feats)

    def fit(self, images: Iterable[np.ndarray], n_images: int, progress: ProgressFn | None = None,
            cancel: CancelFn | None = None) -> None:
        rng = np.random.default_rng(self.config["seed"])
        # One preallocated float16 buffer instead of a list of float32 parts + concatenate (which held the
        # patch bank twice). For 900 images x 160 patches x 384 dims this is ~110 MB instead of ~450 MB.
        bank: np.ndarray | None = None
        n_rows = 0
        embs: list[np.ndarray] = []
        batch: list[np.ndarray] = []
        done = 0
        t0 = time.time()
        ppi = int(self.config["patches_per_image"])

        def flush():
            nonlocal done, bank, n_rows
            patches, _, emb = self._patch_features(batch)
            p = patches.cpu().numpy()
            for row in p:
                k = min(ppi, len(row))
                pick = row[rng.choice(len(row), k, replace=False)].astype(np.float16)
                if bank is None:
                    bank = np.empty((max(n_images, 1) * ppi, pick.shape[1]), np.float16)
                if n_rows + k > len(bank):
                    bank = np.concatenate([bank, np.empty((max(len(bank) // 2, k), bank.shape[1]), np.float16)])
                bank[n_rows:n_rows + k] = pick
                n_rows += k
            embs.append(emb)
            done += len(batch)
            batch.clear()
            if progress:
                progress(0.8 * done / max(n_images, 1), f"Extracted features for {done}/{n_images} images")

        for im in images:
            batch.append(im)
            if len(batch) >= self.config["batch_size"]:
                self._check_cancel(cancel)
                flush()
        if batch:
            flush()
        if bank is None or n_rows == 0:
            raise ValueError("No training images supplied")
        all_feats = bank[:n_rows]
        t_feat = time.time() - t0
        n_select = int(min(self.config["max_memory_bank"], max(1, self.config["coreset_ratio"] * len(all_feats))))
        if progress:
            progress(0.82, f"Coreset selection: {n_select} of {len(all_feats)} patches")
        sel = greedy_coreset(all_feats, n_select, self.config["projection_dim"], self.config["seed"], cancel)
        self._check_cancel(cancel)
        self.memory_bank = np.ascontiguousarray(all_feats[sel], dtype=np.float32)
        n_patches = len(all_feats)
        del bank, all_feats
        emb = np.concatenate(embs)
        self.ood.fit(emb)
        self.history.append({"step": "feature_extraction", "images": done, "patches": n_patches,
                             "seconds": round(t_feat, 2)})
        self.history.append({"step": "coreset", "memory_bank": len(self.memory_bank),
                             "seconds": round(time.time() - t0 - t_feat, 2)})
        if progress:
            progress(1.0, "PatchCore fitted")

    def predict_batch(self, images: list[np.ndarray]) -> list[Prediction]:
        import torch
        import torch.nn.functional as F

        if self.memory_bank is None:
            raise RuntimeError("PatchCore model is not fitted")
        patches, (h, w), emb = self._patch_features(images)
        bank = torch.from_numpy(self.memory_bank).to(patches.device)
        b, p, c = patches.shape
        q = patches.reshape(b * p, c)
        mins = []
        chunk = int(self.config["knn_chunk"])
        with torch.inference_mode():
            for i in range(0, len(q), chunk):
                mins.append(torch.cdist(q[i:i + chunk], bank).min(dim=1).values)
            dist = torch.cat(mins).reshape(b, 1, h, w)
            size = self.config["input_size"]
            up = F.interpolate(dist, size=(size, size), mode="bilinear", align_corners=False)
        maps = up[:, 0].cpu().numpy()
        ood = self.ood.score(emb)
        out = []
        for i in range(b):
            m = smooth_map(maps[i], self.config["blur_sigma"])
            out.append(Prediction(score=float(m.max()), anomaly_map=m, ood_score=float(ood[i])))
        return out

    def state_arrays(self) -> dict[str, np.ndarray]:
        assert self.memory_bank is not None
        return {"memory_bank": self.memory_bank.astype(np.float16), **self.ood.arrays()}

    def load_state(self, arrays: dict[str, np.ndarray]) -> None:
        self.memory_bank = arrays["memory_bank"].astype(np.float32)
        self.ood.load(arrays)
