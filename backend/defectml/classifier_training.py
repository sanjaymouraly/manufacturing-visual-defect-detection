"""Train/evaluate the supervised defect-type classifier attached to an anomaly model.

Protocol: train on anomalous samples of the ``val`` split (pooled with GROUND-TRUTH masks),
evaluate on anomalous samples of the ``test`` split pooled with the anomaly model's PREDICTED
maps (as at inference time). Multi-label samples are excluded (ambiguous target). Requires
genuine defect-type annotations; refuses to run otherwise.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from defectml.data.manifest import Dataset
from defectml.imaging import load_image, load_mask
from defectml.models import load_model
from defectml.models.classifier import DefectTypeClassifier, region_embedding


def _embed(ds: Dataset, samples, maps=None, batch: int = 8) -> np.ndarray:
    out = []
    for i in range(0, len(samples), batch):
        chunk = samples[i:i + batch]
        imgs = [load_image(ds.abspath(s.path)) for s in chunk]
        if maps is None:
            w = [load_mask(ds.abspath(s.mask) if s.mask else None, im.shape[:2]).astype(np.float32)
                 for s, im in zip(chunk, imgs, strict=True)]
        else:
            w = maps[i:i + batch]
        out.append(region_embedding(imgs, w))
    return np.concatenate(out)


def train_defect_classifier(ds: Dataset, model_dir: Path, min_per_class: int = 5,
                            reject_threshold: float = 0.5) -> dict:
    if not ds.info.defect_types_annotated:
        raise ValueError("Dataset has no genuine defect-type annotations; classifier not trained.")
    train = [s for s in ds.split("val") if s.is_anomaly and s.defect_type and "|" not in s.defect_type]
    test = [s for s in ds.split("test") if s.is_anomaly and s.defect_type and "|" not in s.defect_type]
    excluded_multilabel = sum(1 for s in ds.samples if s.is_anomaly and "|" in (s.defect_type or ""))
    counts = Counter(s.defect_type for s in train)
    clf = DefectTypeClassifier(reject_threshold=reject_threshold)
    X = _embed(ds, train)
    info = clf.fit(X, [s.defect_type for s in train], min_per_class=min_per_class)
    test = [s for s in test if s.defect_type in clf.classes]
    result: dict = {"train_counts": dict(counts), "excluded_multilabel": excluded_multilabel, **info}
    if test:
        model, _ = load_model(model_dir)
        maps = [p.anomaly_map for p in model.predict([load_image(ds.abspath(s.path)) for s in test])]
        Xt = _embed(ds, test, maps)
        preds = clf.predict(Xt)
        y_true = [s.defect_type for s in test]
        y_pred = [p["defect_type"] for p in preds]
        known = [i for i, p in enumerate(y_pred) if p != "unknown"]
        P = clf.predict_proba(Xt)
        y_top = [clf.classes[int(j)] for j in P.argmax(1)]
        result["test"] = {
            "n": len(test),
            "test_counts": dict(Counter(y_true)),
            "top1_accuracy": float(np.mean([a == b for a, b in zip(y_true, y_top, strict=True)])),
            "macro_f1_top1": float(f1_score(y_true, y_top, labels=clf.classes, average="macro", zero_division=0)),
            "unknown_rate": 1 - len(known) / len(test),
            "accuracy_when_confident": (float(np.mean([y_true[i] == y_pred[i] for i in known])) if known else None),
            "confusion_matrix": {"labels": clf.classes,
                                 "matrix": confusion_matrix(y_true, y_top, labels=clf.classes).tolist()},
            "protocol": "train=val anomalies (GT-mask pooling); test=test anomalies (predicted-map pooling)",
        }
    clf.save(Path(model_dir) / "classifier")
    (Path(model_dir) / "classifier" / "evaluation.json").write_text(json.dumps(result, indent=2, default=float),
                                                                    encoding="utf-8")
    return result
