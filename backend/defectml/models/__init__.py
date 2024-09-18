"""Model registry: construct, save and load anomaly-detection models."""

from __future__ import annotations

import json
from pathlib import Path

from defectml.imaging import sha256_file
from defectml.models.base import AnomalyModel, Prediction, load_arrays, save_arrays
from defectml.models.padim import PaDiM
from defectml.models.patchcore import PatchCore
from defectml.models.pixelstats import PixelStats

ALGORITHMS: dict[str, type[AnomalyModel]] = {
    "patchcore": PatchCore,
    "padim": PaDiM,
    "pixelstats": PixelStats,
}

ALGORITHM_INFO = {
    "patchcore": "PatchCore - nearest-neighbour memory bank of pretrained ResNet18 patch features",
    "padim": "PaDiM - per-position multivariate Gaussian of pretrained ResNet18 features",
    "pixelstats": "PixelStats baseline - per-pixel Lab colour z-score vs. aligned normal images (no DL)",
}


class ModelLoadError(RuntimeError):
    pass


def build_model(algorithm: str, **config) -> AnomalyModel:
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from {sorted(ALGORITHMS)}")
    return ALGORITHMS[algorithm](**config)


def save_model(model: AnomalyModel, out_dir: Path, meta: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = out_dir / "weights.npz"
    save_arrays(weights, model.state_arrays())
    full = {
        "format_version": 1,
        "algorithm": model.algorithm,
        "config": model.config,
        "history": model.history,
        "weights_sha256": sha256_file(weights),
        **meta,
    }
    (out_dir / "model.json").write_text(json.dumps(full, indent=2, default=float), encoding="utf-8")
    return full


def load_model(model_dir: str | Path, device: str | None = None) -> tuple[AnomalyModel, dict]:
    d = Path(model_dir)
    try:
        meta = json.loads((d / "model.json").read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ModelLoadError(f"Model checkpoint not found at {d}") from exc
    except json.JSONDecodeError as exc:
        raise ModelLoadError(f"Corrupt model.json in {d}") from exc
    weights = d / "weights.npz"
    if not weights.exists():
        raise ModelLoadError(f"weights.npz missing in {d}")
    if meta.get("weights_sha256") and sha256_file(weights) != meta["weights_sha256"]:
        raise ModelLoadError(f"Checksum mismatch for {weights} - artifact corrupt or tampered")
    cfg = dict(meta.get("config", {}))
    if device:
        cfg["device"] = device
    model = build_model(meta["algorithm"], **cfg)
    try:
        model.load_state(load_arrays(weights))
    except (KeyError, ValueError, OSError) as exc:
        raise ModelLoadError(f"Could not load weights from {weights}: {exc}") from exc
    model.history = meta.get("history", [])
    return model, meta


__all__ = ["ALGORITHMS", "ALGORITHM_INFO", "AnomalyModel", "ModelLoadError", "Prediction", "build_model",
           "load_model", "save_model"]
