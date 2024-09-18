"""End-to-end inspection of a single image: quality check -> inference -> localisation -> decision.

Inference contract (also documented in docs/API.md):

input   RGB uint8 image (any size >= 32 px), product's active model directory
output  :class:`InspectionOutput` with raw ``score`` (unitless anomaly score, NOT a probability),
        ``normalized_score`` (= score / fail_threshold; 1.0 = at rejection threshold),
        ``calibrated_probability`` (only when Platt calibration was fitted on validation data),
        ``decision`` PASS/FAIL/REVIEW + reasons, ``severity``, ``regions`` (original pixel coords),
        ``quality`` report, ``ood_score``, optional ``defect_type`` (only if a classifier trained on
        genuine defect-type labels exists), anomaly map and rendered heatmap/overlay.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from defectml.decision import Decision, PolicyConfig, Thresholds, decide
from defectml.evaluation.metrics import platt_apply
from defectml.imaging import QualityLimits, assess_quality
from defectml.localization import extract_regions, render_heatmap, render_overlay
from defectml.models import load_model
from defectml.models.classifier import DefectTypeClassifier, region_embedding


@dataclass
class InspectionOutput:
    score: float | None
    thresholds: dict
    decision: Decision
    calibrated_probability: float | None
    regions: list[dict]
    quality: dict
    ood_score: float | None
    latency_ms: float
    model_name: str
    algorithm: str
    defect_type: dict | None = None
    anomaly_map: np.ndarray | None = None
    heatmap: np.ndarray | None = None
    overlay: np.ndarray | None = None
    mask: np.ndarray | None = None
    secondary: dict | None = None
    extra: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "score": self.score, "thresholds": self.thresholds, **self.decision.to_dict(),
            "calibrated_probability": self.calibrated_probability, "regions": self.regions,
            "quality": self.quality, "ood_score": self.ood_score, "latency_ms": self.latency_ms,
            "model_name": self.model_name, "algorithm": self.algorithm, "defect_type": self.defect_type,
            "secondary": self.secondary,
        }


class Inspector:
    def __init__(self, model_dir: str | Path, device: str | None = None):
        self.model_dir = Path(model_dir)
        self.model, self.meta = load_model(self.model_dir, device=device)
        self.classifier: DefectTypeClassifier | None = None
        if (self.model_dir / "classifier" / "classifier.json").exists():
            self.classifier = DefectTypeClassifier.load(self.model_dir / "classifier")

    def thresholds(self, override: dict | None = None) -> Thresholds:
        t = dict(self.meta["thresholds"])
        for k, v in (override or {}).items():
            if v is not None and k in ("fail_threshold", "pass_threshold", "seg_threshold", "ood_threshold"):
                t[k] = float(v)
        th = Thresholds(t["fail_threshold"], min(t["pass_threshold"], t["fail_threshold"]),
                        t.get("seg_threshold"), t.get("ood_threshold"))
        th.validate()
        return th

    def inspect(self, img: np.ndarray, policy: PolicyConfig | None = None, threshold_override: dict | None = None,
                secondary: Inspector | None = None, quality_limits: QualityLimits | None = None,
                render: bool = True, alpha: float = 0.45) -> InspectionOutput:
        policy = policy or PolicyConfig()
        th = self.thresholds(threshold_override)
        q = assess_quality(img, quality_limits)
        t0 = time.perf_counter()
        pred = self.model.predict_batch([img])[0]
        latency = (time.perf_counter() - t0) * 1000
        regions: list[dict] = []
        mask = None
        if pred.anomaly_map is not None:
            seg_t = th.seg_threshold if th.seg_threshold is not None else th.fail_threshold
            regions, mask = extract_regions(pred.anomaly_map, seg_t, img.shape[:2])
        area = float(sum(r["area_fraction"] for r in regions))
        sec_info = None
        sec_decision = None
        if secondary is not None:
            try:
                sth = secondary.thresholds()
                sp = secondary.model.predict_batch([img])[0]
                sec_decision = "FAIL" if sp.score >= sth.fail_threshold else ("PASS" if sp.score < sth.pass_threshold
                                                                            else "REVIEW")
                sec_info = {"model_name": secondary.meta["name"], "score": sp.score, "decision": sec_decision}
            except Exception as exc:  # secondary failure must not break primary inspection
                sec_info = {"error": str(exc)}
        dec = decide(pred.score, th, quality_issues=q.issues, ood_score=pred.ood_score,
                     secondary_decision=sec_decision, area_fraction=area, cfg=policy)
        cal = self.meta.get("calibration")
        prob = float(platt_apply(cal, pred.score)) if cal else None
        if not regions and pred.anomaly_map is not None and dec.decision in ("FAIL", "REVIEW"):
            # The pixel threshold (F1-optimal on validation masks) can exceed the image score, which would leave a
            # FAIL/REVIEW without any box. For the operator only (decision and metrics are unaffected), mark the
            # peak area instead and flag it as approximate.
            peak_t = th.fail_threshold if dec.decision == "FAIL" else 0.9 * float(pred.score)
            regions, mask = extract_regions(pred.anomaly_map, min(peak_t, float(pred.score)), img.shape[:2],
                                            max_regions=3)
            for r in regions:
                r["approximate"] = True
        dtype = None
        if self.classifier is not None and dec.decision != "PASS":
            emb = region_embedding([img], [pred.anomaly_map], self.model.config["input_size"],
                                   self.model.config.get("device", "auto"))
            dtype = self.classifier.predict(emb)[0]
            dtype["note"] = "Defect-type classifier trained on annotated defect classes; 'unknown' if not confident"
        out = InspectionOutput(
            score=float(pred.score), thresholds={"fail_threshold": th.fail_threshold, "pass_threshold": th.pass_threshold,
                                                 "seg_threshold": th.seg_threshold, "ood_threshold": th.ood_threshold},
            decision=dec, calibrated_probability=prob, regions=regions, quality=q.to_dict(),
            ood_score=pred.ood_score, latency_ms=round(latency, 1), model_name=self.meta["name"],
            algorithm=self.meta["algorithm"], defect_type=dtype, anomaly_map=pred.anomaly_map, mask=mask,
            secondary=sec_info,
        )
        if render and pred.anomaly_map is not None:
            lo = self.meta["thresholds"].get("map_low", 0.0)
            hi = self.meta["thresholds"].get("map_high", th.fail_threshold * 1.5)
            out.heatmap = render_heatmap(pred.anomaly_map, img.shape[:2], lo, hi)
            out.overlay = render_overlay(img, out.heatmap, mask, alpha, regions)
        return out
