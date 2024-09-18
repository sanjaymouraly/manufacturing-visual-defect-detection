"""Pretrained CNN feature extractor shared by PatchCore, PaDiM and the defect-type classifier.

Weights: torchvision ResNet18 ``IMAGENET1K_V1`` downloaded at runtime by torchvision into
``$TORCH_HOME`` (not redistributed by this repository). Pretrained ImageNet weights may carry
their own terms of use; review them before commercial deployment (see docs/MODEL_CARD.md).
"""

from __future__ import annotations

import os
import threading

import numpy as np

from defectml.imaging import resize_rgb

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

_lock = threading.Lock()
_cache: dict[tuple[str, str], object] = {}


class BackboneUnavailable(RuntimeError):
    """Raised when torch/torchvision or the pretrained weights cannot be loaded."""


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except Exception:
        return False
    return True


def resolve_device(device: str = "auto") -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def configure_threads() -> None:
    import torch

    n = int(os.environ.get("DL_TORCH_THREADS", "0") or 0)
    if n > 0:
        torch.set_num_threads(n)


class FeatureExtractor:
    """Returns intermediate ResNet feature maps for a batch of RGB uint8 images."""

    def __init__(self, arch: str = "resnet18", device: str = "auto", input_size: int = 224):
        if not torch_available():
            raise BackboneUnavailable("PyTorch/torchvision are not installed")
        import torch
        import torchvision

        configure_threads()
        self.device = resolve_device(device)
        self.input_size = input_size
        self.arch = arch
        try:
            if arch == "resnet18":
                net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            elif arch == "wide_resnet50_2":
                net = torchvision.models.wide_resnet50_2(
                    weights=torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
            else:
                raise BackboneUnavailable(f"Unsupported backbone {arch}")
        except BackboneUnavailable:
            raise
        except Exception as exc:  # network failures, corrupt cache ...
            raise BackboneUnavailable(
                f"Could not load pretrained {arch} weights ({exc.__class__.__name__}). "
                "Run 'python -m defectml.cli fetch-weights' with internet access once, or set TORCH_HOME "
                "to a directory containing the cached weights."
            ) from exc
        net.eval().to(self.device)
        for p in net.parameters():
            p.requires_grad_(False)
        self.net = net
        self._torch = torch

    def preprocess(self, images: list[np.ndarray]):
        torch = self._torch
        arr = np.stack([resize_rgb(im, self.input_size) for im in images]).astype(np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(arr.transpose(0, 3, 1, 2).copy()).to(self.device)

    def __call__(self, images: list[np.ndarray]) -> dict:
        torch = self._torch
        x = self.preprocess(images)
        n = self.net
        with torch.inference_mode():
            x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
            l1 = n.layer1(x)
            l2 = n.layer2(l1)
            l3 = n.layer3(l2)
        return {"layer1": l1, "layer2": l2, "layer3": l3}

    def global_embedding(self, feats: dict) -> np.ndarray:
        """Image-level embedding (avg-pooled layer2+layer3) used for OOD detection & classification."""
        torch = self._torch
        g = torch.cat([feats["layer2"].mean(dim=(2, 3)), feats["layer3"].mean(dim=(2, 3))], dim=1)
        return g.cpu().numpy().astype(np.float32)


def get_extractor(arch: str = "resnet18", device: str = "auto", input_size: int = 224) -> FeatureExtractor:
    key = (arch, f"{device}:{input_size}")
    with _lock:
        if key not in _cache:
            _cache[key] = FeatureExtractor(arch, device, input_size)
        return _cache[key]  # type: ignore[return-value]
