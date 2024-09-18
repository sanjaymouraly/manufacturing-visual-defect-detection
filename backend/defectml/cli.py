"""DefectLens ML command-line interface.

Datasets and models default to the same directories the web app uses (<DL_VAR_DIR>/datasets and
<DL_VAR_DIR>/models, honouring .env), and DATASET/MODEL arguments accept a path or a bare name found there.

Examples::

    defectml fetch-weights
    defectml gen-synthetic
    defectml prepare-visa --download --accept-license --categories pcb1 capsules cashew
    defectml validate visa-pcb1
    defectml train visa-pcb1 --algorithm patchcore                # -> <models>/visa-pcb1-patchcore
    defectml evaluate visa-pcb1 --models visa-pcb1-patchcore visa-pcb1-padim --out reports/visa-pcb1
    defectml robustness visa-pcb1 --model visa-pcb1-patchcore --severities 1 3 5
    defectml inspect samples/demo-pcb/N_0000_normal.png --model demo-pcb-patchcore-demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _progress(frac: float, msg: str) -> None:
    print(f"\r[{frac * 100:5.1f}%] {msg[:90]:<90}", end="", file=sys.stderr, flush=True)
    if frac >= 1.0:
        print(file=sys.stderr)


def cmd_fetch_weights(a):
    from defectml.models.backbone import get_extractor

    get_extractor("resnet18")
    print("ResNet18 ImageNet weights available.")


def cmd_gen_synthetic(a):
    from defectml.data.synthetic import generate_all

    for p in generate_all(Path(a.out), seed=a.seed, n_normal=a.n_normal, n_anomaly=a.n_anomaly):
        print(f"generated {p}")


def cmd_prepare_visa(a):
    from defectml.data import visa

    raw = Path(a.raw) if a.raw else None
    sha = ""
    if a.tar:
        if not a.accept_license:
            raise SystemExit("VisA is CC BY 4.0 - pass --accept-license to confirm attribution obligations.")
        tar = Path(a.tar)
        if a.verify:
            from defectml.imaging import sha256_file

            sha = sha256_file(tar)
            if sha != visa.VISA_SHA256:
                raise SystemExit(f"Checksum mismatch: {sha}")
        raw = Path(a.out) / "_visa_raw"
        visa.extract_categories(tar, raw, a.categories)
    elif a.download:
        tar = visa.download_visa(Path(a.out) / "_downloads", a.accept_license)
        sha = visa.VISA_SHA256
        raw = Path(a.out) / "_visa_raw"
        visa.extract_categories(tar, raw, a.categories)
    if raw is None:
        raise SystemExit("Provide --raw, --tar or --download")
    split_csv = Path(a.split_csv) if a.split_csv else None
    if split_csv is None or not split_csv.exists():
        import urllib.request

        split_csv = Path(a.out) / "_visa_1cls.csv"
        urllib.request.urlretrieve(visa.VISA_SPLIT_URL, split_csv)  # noqa: S310
    for c in a.categories:
        out = visa.prepare_visa_category(raw, split_csv, c, Path(a.out), max_side=a.max_side, archive_sha256=sha)
        print(f"prepared {out}")


def cmd_import_folder(a):
    from defectml.data.folder import import_folder

    out, skipped = import_folder(Path(a.src), Path(a.out), a.name, a.product or a.name, a.license)
    print(f"imported {out}; skipped {len(skipped)}")


def cmd_validate(a):
    from defectml.data.manifest import Dataset
    from defectml.data.validation import validate_dataset

    r = validate_dataset(Dataset(a.dataset))
    keep = {k: r[k] for k in ("dataset", "num_samples", "labels", "splits", "defect_types", "near_duplicate_pairs",
                              "warnings", "valid")}
    keep["leakage_free"] = r["leakage"]["leakage_free"]
    keep["errors"] = len(r["errors"])
    keep["mask_issues"] = len(r["mask_issues"])
    print(json.dumps(keep, indent=2))


def cmd_train(a):
    from defectml.data.manifest import Dataset
    from defectml.training import train_on_dataset

    cfg = json.loads(a.config) if a.config else {}
    meta = train_on_dataset(Dataset(a.dataset), a.algorithm, Path(a.out), name=a.name, config=cfg,
                            max_train_images=a.max_train_images, seed=a.seed, progress=_progress)
    v = meta["validation_metrics"]
    print(json.dumps({"model": meta["name"], "thresholds": meta["thresholds"], "val_auroc": v["ranking"]["auroc"],
                      "fit_seconds": meta["training"]["fit_seconds"]}, indent=2))


def cmd_train_classifier(a):
    from defectml.classifier_training import train_defect_classifier
    from defectml.data.manifest import Dataset

    r = train_defect_classifier(Dataset(a.dataset), Path(a.model_dir), min_per_class=a.min_per_class)
    print(json.dumps(r, indent=2, default=float))


def cmd_evaluate(a):
    from defectml.data.manifest import Dataset
    from defectml.evaluation.benchmark import compare, evaluate_model

    ds = Dataset(a.dataset)
    results = []
    for m in a.models:
        print(f"evaluating {m}", file=sys.stderr)
        results.append(evaluate_model(Path(m), ds, a.split, progress=_progress))
    s = compare(results, Path(a.out) if a.out else None)
    for r in s["rows"]:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}))


def cmd_robustness(a):
    from defectml.data.manifest import Dataset
    from defectml.evaluation.robustness import run_robustness

    r = run_robustness(Path(a.model), Dataset(a.dataset), a.degradations, a.severities, max_samples=a.max_samples,
                       out_dir=Path(a.out) if a.out else None, progress=_progress)
    print(json.dumps({"clean": r["clean"], "robustness_score": r["robustness_score"],
                      "worst": r["worst_conditions"][:3]}, indent=2, default=float))


def cmd_domain_shift(a):
    from defectml.data.manifest import Dataset
    from defectml.evaluation.robustness import domain_shift_experiment

    r = domain_shift_experiment(Path(a.model), Dataset(a.dataset), Path(a.out) if a.out else None,
                                max_samples=a.max_samples)
    print(json.dumps(r, indent=2, default=float))


def cmd_export(a):
    from defectml.data.manifest import Dataset
    from defectml.export import export_onnx
    from defectml.imaging import load_image

    imgs = None
    if a.dataset:
        ds = Dataset(a.dataset)
        imgs = [load_image(ds.abspath(s.path)) for s in ds.split("test")[:4]]
    print(json.dumps({k: v for k, v in export_onnx(Path(a.model), verify_images=imgs).items()
                      if k not in ("torch_scores", "onnx_scores")}, indent=2))


def cmd_inspect(a):
    from defectml.imaging import load_image
    from defectml.inference import Inspector

    out = Inspector(a.model).inspect(load_image(a.image), render=False)
    print(json.dumps(out.summary(), indent=2, default=float))


def _storage_dirs() -> tuple[Path, Path]:
    """Datasets/models directories exactly as the web app sees them (DL_VAR_DIR, DL_DATASETS_DIR, .env, ...)."""
    try:
        from app.config import get_settings

        s = get_settings()
        return Path(s.datasets_dir), Path(s.models_dir)
    except ImportError:  # defectml used standalone, without the app package
        import os

        var = Path(os.environ.get("DL_VAR_DIR") or Path(__file__).resolve().parents[2] / "var")
        return (Path(os.environ.get("DL_DATASETS_DIR") or var / "datasets"),
                Path(os.environ.get("DL_MODELS_DIR") or var / "models"))


def _resolve(arg: str, base: Path) -> str:
    """Accept a path, or a bare name of a dataset/model registered under the app's storage directory."""
    p = Path(arg)
    if not p.exists() and len(p.parts) == 1 and (base / arg).exists():
        return str(base / arg)
    return arg


def _normalise_args(a) -> None:
    ds_dir, models_dir = _storage_dirs()
    if a.cmd in ("gen-synthetic", "prepare-visa", "import-folder") and not a.out:
        a.out = str(ds_dir)
    if getattr(a, "dataset", None):
        a.dataset = _resolve(a.dataset, ds_dir)
    for attr in ("model", "model_dir"):
        if getattr(a, attr, None):
            setattr(a, attr, _resolve(getattr(a, attr), models_dir))
    if getattr(a, "models", None):
        a.models = [_resolve(m, models_dir) for m in a.models]
    if a.cmd == "train" and not a.out:
        a.out = str(models_dir / f"{Path(a.dataset).name}-{a.algorithm}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="defectml", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("fetch-weights").set_defaults(fn=cmd_fetch_weights)
    p = sp.add_parser("gen-synthetic")
    p.add_argument("--out", help="datasets directory (default: the app's datasets dir, <DL_VAR_DIR>/datasets)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-normal", type=int, default=160)
    p.add_argument("--n-anomaly", type=int, default=60)
    p.set_defaults(fn=cmd_gen_synthetic)
    p = sp.add_parser("prepare-visa")
    p.add_argument("--raw", help="directory with extracted VisA categories")
    p.add_argument("--tar", help="path to VisA_20220922.tar")
    p.add_argument("--download", action="store_true", help="download the archive from the official host")
    p.add_argument("--split-csv", help="official split_csv/1cls.csv (downloaded if missing)")
    p.add_argument("--categories", nargs="+", default=["pcb1"])
    p.add_argument("--out", help="datasets directory (default: the app's datasets dir, <DL_VAR_DIR>/datasets)")
    p.add_argument("--max-side", type=int, default=512)
    p.add_argument("--accept-license", action="store_true")
    p.add_argument("--no-verify", dest="verify", action="store_false")
    p.set_defaults(fn=cmd_prepare_visa)
    p = sp.add_parser("import-folder")
    p.add_argument("src")
    p.add_argument("--name", required=True)
    p.add_argument("--product")
    p.add_argument("--license", default="user-provided")
    p.add_argument("--out", help="datasets directory (default: the app's datasets dir, <DL_VAR_DIR>/datasets)")
    p.set_defaults(fn=cmd_import_folder)
    p = sp.add_parser("validate")
    p.add_argument("dataset")
    p.set_defaults(fn=cmd_validate)
    p = sp.add_parser("train")
    p.add_argument("dataset")
    p.add_argument("--algorithm", default="patchcore", choices=["patchcore", "padim", "pixelstats"])
    p.add_argument("--out", help="model directory (default: <models dir>/<dataset>-<algorithm>)")
    p.add_argument("--name")
    p.add_argument("--config", help="JSON dict of model config overrides")
    p.add_argument("--max-train-images", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_train)
    p = sp.add_parser("train-classifier")
    p.add_argument("dataset")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--min-per-class", type=int, default=5)
    p.set_defaults(fn=cmd_train_classifier)
    p = sp.add_parser("evaluate")
    p.add_argument("dataset")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_evaluate)
    p = sp.add_parser("robustness")
    p.add_argument("dataset")
    p.add_argument("--model", required=True)
    p.add_argument("--degradations", nargs="*")
    p.add_argument("--severities", nargs="*", type=int)
    p.add_argument("--max-samples", type=int, default=120)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_robustness)
    p = sp.add_parser("domain-shift")
    p.add_argument("dataset")
    p.add_argument("--model", required=True)
    p.add_argument("--max-samples", type=int, default=150)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_domain_shift)
    p = sp.add_parser("export-onnx")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", help="dataset used to verify ONNX/PyTorch parity")
    p.set_defaults(fn=cmd_export)
    p = sp.add_parser("inspect")
    p.add_argument("image")
    p.add_argument("--model", required=True)
    p.set_defaults(fn=cmd_inspect)
    a = ap.parse_args(argv)
    _normalise_args(a)
    a.fn(a)


if __name__ == "__main__":
    main()
