"""Supervised defect-type classifier (transfer learning, linear probe with early stopping).

Only trained when the dataset carries genuine defect-type annotations
(``DatasetInfo.defect_types_annotated``) and each class has at least ``min_per_class`` samples.

Features: ResNet18 layer2+layer3 activations pooled with weights given by the ground-truth mask
(during training) or the anomaly map (at inference), so the classifier focuses on the defect
region. A softmax head is trained with class-balanced cross-entropy, weight decay and early
stopping on a validation subset. Predictions whose max probability falls below
``reject_threshold`` are reported as ``unknown`` (open-set behaviour) - unrecognised defects are
handled by the anomaly detector, never force-fitted into a known class.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from defectml.models.backbone import get_extractor
from defectml.models.base import load_arrays, save_arrays


def region_embedding(images: list[np.ndarray], weights: Sequence[np.ndarray | None], input_size: int = 224,
                     device: str = "auto") -> np.ndarray:
    import torch
    import torch.nn.functional as F

    ext = get_extractor("resnet18", device, input_size)
    feats = ext(images)
    l2 = feats["layer2"]
    l3 = F.interpolate(feats["layer3"], size=l2.shape[-2:], mode="bilinear", align_corners=False)
    f = torch.cat([l2, l3], dim=1)  # B, C, 28, 28
    h, w = f.shape[-2:]
    out = []
    for i, wmap in enumerate(weights):
        if wmap is None or float(np.max(wmap)) <= 0:
            wt = torch.ones((h, w))
        else:
            wm = cv2.resize(wmap.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
            wm = wm - float(np.min(wm))
            wt = torch.from_numpy(wm / (float(np.max(wm)) + 1e-6))
        wt = wt.to(f.device)
        pooled_region = (f[i] * wt).sum(dim=(1, 2)) / (wt.sum() + 1e-6)
        pooled_global = f[i].mean(dim=(1, 2))
        out.append(torch.cat([pooled_region, pooled_global]).cpu().numpy())
    return np.array(out, np.float32)


class DefectTypeClassifier:
    def __init__(self, classes: list[str] | None = None, reject_threshold: float = 0.5):
        self.classes = classes or []
        self.reject_threshold = reject_threshold
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self.mu: np.ndarray | None = None
        self.sigma: np.ndarray | None = None
        self.history: list[dict] = []

    def fit(self, X: np.ndarray, y_labels: list[str], min_per_class: int = 5, max_epochs: int = 400,
            patience: int = 30, lr: float = 0.01, weight_decay: float = 1e-3, seed: int = 0) -> dict:
        import torch

        counts = Counter(y_labels)
        self.classes = sorted(c for c, n in counts.items() if n >= min_per_class)
        if len(self.classes) < 2:
            raise ValueError(f"Need >=2 defect classes with >={min_per_class} samples each; got {dict(counts)}")
        keep = [i for i, c in enumerate(y_labels) if c in self.classes]
        X = X[keep]
        y = np.array([self.classes.index(y_labels[i]) for i in keep])
        self.mu, self.sigma = X.mean(0), X.std(0) + 1e-6
        Xn = (X - self.mu) / self.sigma
        rng = np.random.default_rng(seed)
        # stratified 80/20 split for early stopping
        val_idx: list[int] = []
        for c in range(len(self.classes)):
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            val_idx.extend(idx[: max(1, len(idx) // 5)])
        val_mask = np.zeros(len(y), bool)
        val_mask[val_idx] = True
        torch.manual_seed(seed)
        Xt, yt = torch.from_numpy(Xn[~val_mask]), torch.from_numpy(y[~val_mask])
        Xv, yv = torch.from_numpy(Xn[val_mask]), torch.from_numpy(y[val_mask])
        cw = torch.tensor([len(y) / (len(self.classes) * max(1, (y == c).sum())) for c in range(len(self.classes))],
                          dtype=torch.float32)
        head = torch.nn.Linear(X.shape[1], len(self.classes))
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
        lossf = torch.nn.CrossEntropyLoss(weight=cw)
        best, best_state, bad = float("inf"), None, 0
        for epoch in range(max_epochs):
            head.train()
            opt.zero_grad()
            loss = lossf(head(Xt), yt)
            loss.backward()
            opt.step()
            head.eval()
            with torch.no_grad():
                vl = float(lossf(head(Xv), yv))
                vacc = float((head(Xv).argmax(1) == yv).float().mean())
            if epoch % 10 == 0:
                self.history.append({"epoch": epoch, "train_loss": round(float(loss), 4), "val_loss": round(vl, 4),
                                     "val_acc": round(vacc, 4)})
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in head.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    self.history.append({"epoch": epoch, "early_stopped": True, "best_val_loss": round(best, 4)})
                    break
        assert best_state is not None
        head.load_state_dict(best_state)
        self.W = head.weight.detach().numpy().T.copy()
        self.b = head.bias.detach().numpy().copy()
        return {"classes": self.classes, "train_counts": {c: counts[c] for c in self.classes},
                "excluded_classes": {c: n for c, n in counts.items() if c not in self.classes},
                "best_val_loss": round(best, 4)}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.W is not None
        z = ((X - self.mu) / self.sigma) @ self.W + self.b
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def predict(self, X: np.ndarray) -> list[dict]:
        P = self.predict_proba(X)
        out = []
        for p in P:
            j = int(p.argmax())
            label = self.classes[j] if p[j] >= self.reject_threshold else "unknown"
            out.append({"defect_type": label, "probability": round(float(p[j]), 4),
                        "top": {c: round(float(v), 4) for c, v in sorted(zip(self.classes, p, strict=True),
                                                                         key=lambda t: -t[1])[:3]}})
        return out

    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        assert self.W is not None and self.b is not None and self.mu is not None and self.sigma is not None
        save_arrays(d / "classifier.npz", {"W": self.W, "b": self.b, "mu": self.mu, "sigma": self.sigma})
        (d / "classifier.json").write_text(json.dumps({"classes": self.classes, "reject_threshold": self.reject_threshold,
                                                       "history": self.history}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, d: Path) -> DefectTypeClassifier:
        meta = json.loads((d / "classifier.json").read_text(encoding="utf-8"))
        c = cls(meta["classes"], meta["reject_threshold"])
        a = load_arrays(d / "classifier.npz")
        c.W, c.b, c.mu, c.sigma = a["W"], a["b"], a["mu"], a["sigma"]
        c.history = meta.get("history", [])
        return c
