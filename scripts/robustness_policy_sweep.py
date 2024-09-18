#!/usr/bin/env python
"""Count full-policy decisions (PASS / REVIEW / FAIL) for every robustness degradation.

`defectml robustness` reports AUROC/FRR/recall at the fail threshold. This script answers the question that
matters for escapes: would the *complete* decision policy (pass + fail thresholds, OOD and quality checks)
auto-PASS a defective part under each corruption? It runs every degraded image through the same `Inspector`
the app uses, with the same per-image seeds as the benchmark, so the images are identical.

Usage (from backend/, venv active, DL_VAR_DIR / TORCH_HOME exported as for the app):
    python ../scripts/robustness_policy_sweep.py visa-pcb1 visa-pcb1-patchcore --max-samples 40 --out sweep.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from defectml.cli import _resolve, _storage_dirs
from defectml.data.manifest import Dataset
from defectml.evaluation import degradations as D
from defectml.evaluation.robustness import _subset
from defectml.imaging import load_image
from defectml.inference import Inspector


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("dataset", help="dataset path or registered name")
    p.add_argument("model", help="model path or registered name")
    p.add_argument("--max-samples", type=int, default=40)
    p.add_argument("--severities", type=int, nargs="*", default=[1, 3, 5])
    p.add_argument("--degradations", nargs="*", default=None)
    p.add_argument("--out", type=Path)
    a = p.parse_args()

    ds_dir, model_dir = _storage_dirs()
    ds = Dataset(_resolve(a.dataset, ds_dir))
    inspector = Inspector(_resolve(a.model, model_dir))
    images = [(s.label, load_image(ds.root / s.path)) for s in _subset(ds.split("test"), a.max_samples)]

    results: dict[str, dict[str, int]] = {}

    def run(name: str, transform) -> None:
        c: collections.Counter[str] = collections.Counter()
        for label, img in images:
            x = transform(img) if transform else img
            c[f"{label}|{inspector.inspect(x, render=False).summary()['decision']}"] += 1
        results[name] = dict(sorted(c.items()))
        print(name, results[name], flush=True)

    run("clean", None)
    for d in a.degradations or list(D.DEGRADATIONS):
        for sev in a.severities:

            def tf(img, _d=d, _s=sev):
                seed = int(img[::37, ::41].sum()) % (2**31)  # same per-image seed as run_robustness
                return D.apply(_d, img, None, _s, seed=seed)[0]

            run(f"{d}@{sev}", tf)

    degraded = {k: v for k, v in results.items() if k != "clean"}
    n_anom = sum(1 for lab, _ in images if lab != "normal")
    passed = sum(v.get("anomaly|PASS", 0) for v in degraded.values())
    print(
        f"\ndefective images auto-PASSed under degradation: {passed} / {n_anom * len(degraded)} decisions "
        f"({len(degraded)} conditions); clean: {results['clean'].get('anomaly|PASS', 0)} / {n_anom}"
    )
    if a.out:
        a.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
