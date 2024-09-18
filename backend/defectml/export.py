"""Export fitted PatchCore / PaDiM models as self-contained ONNX graphs.

The exported graph takes ``input`` float32 [N,3,224,224] (ImageNet-normalised RGB) and returns
``anomaly_map`` [N,224,224] and ``score`` [N] on the same scale as the PyTorch implementation,
so the thresholds in model.json remain valid. Parity is verified with onnxruntime after export.
OpenVINO can consume the ONNX file directly (``ovc model.onnx``) - not validated in this repository.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from defectml.models import load_model


def _gauss_kernel(sigma: float):
    import torch

    radius = int(4.0 * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return torch.from_numpy(k.astype(np.float32)), radius


def build_export_module(model):
    import torch
    import torch.nn.functional as F

    ext = model._extractor()
    net = ext.net
    algo = model.algorithm
    size = model.config["input_size"]
    k1d, radius = _gauss_kernel(model.config["blur_sigma"])

    class Exported(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net
            self.register_buffer("kx", k1d.view(1, 1, 1, -1))
            self.register_buffer("ky", k1d.view(1, 1, -1, 1))
            # scipy.ndimage.gaussian_filter (reference path) pads with mode="reflect" = half-sample SYMMETRIC
            # (d c b a | a b c d). torch's F.pad "reflect" omits the edge sample, so pad via a gather index instead.
            r = radius
            pad_idx = list(range(r - 1, -1, -1)) + list(range(size)) + list(range(size - 1, size - 1 - r, -1))
            self.register_buffer("pad_idx", torch.tensor([min(max(i, 0), size - 1) for i in pad_idx], dtype=torch.long))
            if algo == "patchcore":
                bank = torch.from_numpy(model.memory_bank)
                self.register_buffer("bank", bank)
                self.register_buffer("bank_sq", (bank ** 2).sum(1))
            else:
                self.register_buffer("mean", torch.from_numpy(model.mean))
                self.register_buffer("inv_cov", torch.from_numpy(model.inv_cov))
                self.register_buffer("idx", torch.from_numpy(model.channel_idx))

        def blur(self, m):
            m = F.conv2d(m.index_select(3, self.pad_idx), self.kx)
            return F.conv2d(m.index_select(2, self.pad_idx), self.ky)

        def forward(self, x):
            n = self.net
            x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
            l1 = n.layer1(x)
            l2 = n.layer2(l1)
            l3 = n.layer3(l2)
            if algo == "patchcore":
                l3u = F.interpolate(l3, size=l2.shape[-2:], mode="bilinear", align_corners=False)
                f = F.avg_pool2d(torch.cat([l2, l3u], 1), 3, stride=1, padding=1)
                b, c, h, w = f.shape
                q = f.permute(0, 2, 3, 1).reshape(b * h * w, c)
                # the matmul expansion is fast but suffers float32 cancellation when the nearest distance is
                # small relative to the feature norms; use it only to pick the neighbour, then compute the exact
                # Euclidean distance to it (matches torch.cdist in the reference implementation)
                d2 = self.bank_sq[None, :] - 2 * q @ self.bank.T
                nn = self.bank.index_select(0, d2.argmin(dim=1))
                d = torch.sqrt(((q - nn) ** 2).sum(1)).reshape(b, 1, h, w)
            else:
                g = model.config["grid"]
                f = torch.cat([F.adaptive_avg_pool2d(l1, g), l2,
                               F.interpolate(l3, size=(g, g), mode="bilinear", align_corners=False)], 1)
                f = f.index_select(1, self.idx)
                b, c, h, w = f.shape
                e = f.permute(0, 2, 3, 1).reshape(b, h * w, c) - self.mean
                m = torch.einsum("bpi,pij,bpj->bp", e, self.inv_cov, e)
                d = torch.sqrt(torch.clamp(m, min=0)).reshape(b, 1, h, w)
            up = F.interpolate(d, size=(size, size), mode="bilinear", align_corners=False)
            amap = self.blur(up)[:, 0]
            return amap, amap.flatten(1).max(dim=1).values

    return Exported().eval(), ext


def export_onnx(model_dir: Path, out_path: Path | None = None, verify_images: list[np.ndarray] | None = None) -> dict:
    import torch

    model, _meta = load_model(model_dir, device="cpu")
    if model.algorithm not in ("patchcore", "padim"):
        raise ValueError(f"ONNX export supports patchcore/padim, not {model.algorithm}")
    module, ext = build_export_module(model)
    out_path = out_path or Path(model_dir) / "model.onnx"
    size = model.config["input_size"]
    dummy = torch.zeros(1, 3, size, size)
    torch.onnx.export(module, (dummy,), str(out_path), input_names=["input"], output_names=["anomaly_map", "score"],
                      dynamic_axes={"input": {0: "batch"}, "anomaly_map": {0: "batch"}, "score": {0: "batch"}},
                      opset_version=17, dynamo=False)
    report = {"onnx_path": str(out_path), "opset": 17, "input": [None, 3, size, size],
              "preprocessing": "resize to input size, RGB/255, ImageNet mean/std normalisation, NCHW"}
    if verify_images:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        x = ext.preprocess(verify_images).cpu().numpy()
        _, onnx_scores = sess.run(None, {"input": x})
        torch_scores = np.array([p.score for p in model.predict_batch(verify_images)])
        rel = float(np.max(np.abs(onnx_scores - torch_scores) / np.maximum(np.abs(torch_scores), 1e-6)))
        report.update({"parity_max_relative_error": rel, "parity_ok": rel < 1e-3,
                       "torch_scores": torch_scores.tolist(), "onnx_scores": onnx_scores.tolist()})
    (Path(model_dir) / "onnx_export.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
