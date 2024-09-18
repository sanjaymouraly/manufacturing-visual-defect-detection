"""End-to-end training of the deep anomaly models (PatchCore, PaDiM) + ONNX export parity.

Marked ``slow``: needs PyTorch and downloads ImageNet ResNet-18 weights (~45 MB, BSD-3 torchvision) on first
run. Run with ``pytest -m slow``. CI runs these in a separate job. Uses a small SYNTHETIC dataset, so the
AUROC bounds below are sanity checks of the pipeline, not claims about real-world accuracy.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def synth_ds(tmp_path_factory):
    from defectml.data.manifest import Dataset
    from defectml.data.synthetic import generate_product

    root = generate_product("demo-pcb", tmp_path_factory.mktemp("synth"), n_normal=60, n_anomaly=24, n_batches=4, seed=11)
    return Dataset(root)


@pytest.mark.parametrize("algorithm", ["patchcore", "padim"])
def test_train_evaluate_infer_export(synth_ds, tmp_path, algorithm):
    from defectml.export import export_onnx
    from defectml.inference import Inspector
    from defectml.training import train_on_dataset

    out = tmp_path / algorithm
    train_on_dataset(synth_ds, algorithm, out, name=f"test-{algorithm}", seed=0)

    meta = json.loads((out / "model.json").read_text())
    th = meta["thresholds"]
    assert 0 < th["pass_threshold"] <= th["fail_threshold"]
    assert "Youden" in th["method"]

    from defectml.evaluation.benchmark import evaluate_model

    tm = evaluate_model(out, synth_ds, "test")  # held-out split, reporting only
    assert tm["ranking"]["auroc"] >= 0.85, tm["ranking"]  # synthetic defects are easy; this guards regressions
    assert tm["pixel"]["pixel_auroc"] >= 0.85

    insp = Inspector(out, device="cpu")
    test = synth_ds.split("test")
    from defectml.imaging import decode_image_bytes

    def run(samples):
        return [insp.inspect(decode_image_bytes(synth_ds.abspath(s.path).read_bytes())) for s in samples]

    ra = run([s for s in test if s.is_anomaly][:5])
    rn = run([s for s in test if not s.is_anomaly][:5])
    # single pairs can overlap (AUROC < 1); the typical anomaly must still score above the typical normal
    assert np.median([r.score for r in ra]) > np.median([r.score for r in rn])
    for r in ra + rn:
        assert r.anomaly_map is not None and np.isfinite(r.anomaly_map).all()
        assert r.decision.decision in ("PASS", "FAIL", "REVIEW")

    imgs = [decode_image_bytes(synth_ds.abspath(s.path).read_bytes()) for s in test[:3]]
    rep = export_onnx(out, verify_images=imgs)
    assert rep["parity_ok"] and rep["parity_max_relative_error"] < 1e-4, rep  # incl. border peaks
