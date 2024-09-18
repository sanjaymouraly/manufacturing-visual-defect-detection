"""Conservative, configurable accept / reject / review decision policy.

The model output is decision SUPPORT. The policy is designed to fail safe:

* PASS   only when the image is valid, quality checks pass, the image is in-distribution,
         no model disagreement exists and the anomaly score is below ``pass_threshold``
* FAIL   ("suspected defect") when the score is at or above ``fail_threshold``
* REVIEW for everything else: borderline scores, poor image quality, out-of-distribution
         inputs, disagreement between models, or when inspection could not be performed.

An item is never auto-approved when inference failed or no model is available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

PASS, FAIL, REVIEW = "PASS", "FAIL", "REVIEW"
SEVERITIES = ["none", "low", "medium", "high", "critical"]


@dataclass
class Thresholds:
    fail_threshold: float
    pass_threshold: float
    seg_threshold: float | None = None
    ood_threshold: float | None = None

    def validate(self) -> None:
        if self.pass_threshold > self.fail_threshold:
            raise ValueError("pass_threshold must be <= fail_threshold")


@dataclass
class PolicyConfig:
    review_on_quality_issues: bool = True
    review_on_ood: bool = True
    review_on_disagreement: bool = True
    # FAIL decisions with these severities are additionally queued for human confirmation
    review_fail_severities: list[str] = field(default_factory=lambda: ["critical"])
    # severity boundaries on the normalised score (score / fail_threshold) and defect area fraction
    medium_ratio: float = 1.0
    high_ratio: float = 1.5
    critical_ratio: float = 2.5
    high_area_fraction: float = 0.02
    critical_area_fraction: float = 0.10

    @classmethod
    def from_dict(cls, d: dict | None) -> PolicyConfig:
        if not d:
            return cls()
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Decision:
    decision: str
    reasons: list[str]
    severity: str
    normalized_score: float | None
    needs_review: bool

    def to_dict(self) -> dict:
        return asdict(self)


def severity_for(score: float, th: Thresholds, area_fraction: float, cfg: PolicyConfig) -> str:
    if score < th.pass_threshold:
        return "none"
    r = score / th.fail_threshold if th.fail_threshold > 0 else float("inf")
    if r >= cfg.critical_ratio or area_fraction >= cfg.critical_area_fraction:
        return "critical" if r >= cfg.medium_ratio else "high"
    if r >= cfg.high_ratio or (r >= cfg.medium_ratio and area_fraction >= cfg.high_area_fraction):
        return "high"
    if r >= cfg.medium_ratio:
        return "medium"
    return "low"


def decide(score: float | None, th: Thresholds | None, *, quality_issues: list[str] | None = None,
           ood_score: float | None = None, secondary_decision: str | None = None,
           area_fraction: float = 0.0, cfg: PolicyConfig | None = None, error: str | None = None) -> Decision:
    cfg = cfg or PolicyConfig()
    quality_issues = quality_issues or []
    if error is not None or score is None or th is None:
        return Decision(REVIEW, [f"Inspection could not be performed: {error or 'no model available'}"],
                        "none", None, True)
    if score != score:  # NaN guard
        return Decision(REVIEW, ["Model returned an invalid score (NaN)"], "none", None, True)
    reasons: list[str] = []
    norm = round(score / th.fail_threshold, 4) if th.fail_threshold > 0 else None
    sev = severity_for(score, th, area_fraction, cfg)
    if score >= th.fail_threshold:
        reasons.append(f"Anomaly score {score:.4g} >= rejection threshold {th.fail_threshold:.4g}")
        if quality_issues:
            reasons.append("Image quality issues: " + "; ".join(quality_issues))
        needs = sev in cfg.review_fail_severities
        if secondary_decision == PASS and cfg.review_on_disagreement:
            reasons.append("Secondary model disagrees (PASS)")
            needs = True
        if needs and sev in cfg.review_fail_severities:
            reasons.append(f"Severity '{sev}' requires inspector confirmation")
        return Decision(FAIL, reasons, sev, norm, needs)
    flags: list[str] = []
    if score >= th.pass_threshold:
        flags.append(f"Borderline score {score:.4g} between pass ({th.pass_threshold:.4g}) and "
                     f"reject ({th.fail_threshold:.4g}) thresholds")
    if quality_issues and cfg.review_on_quality_issues:
        flags.append("Image quality insufficient: " + "; ".join(quality_issues))
    if (cfg.review_on_ood and ood_score is not None and th.ood_threshold is not None
            and ood_score > th.ood_threshold):
        flags.append(f"Out-of-distribution input (OOD score {ood_score:.3g} > {th.ood_threshold:.3g}); "
                     "unseen conditions")
    if cfg.review_on_disagreement and secondary_decision == FAIL:
        flags.append("Secondary model disagrees (FAIL)")
    if flags:
        return Decision(REVIEW, flags, sev if sev != "none" else "low", norm, True)
    return Decision(PASS, [f"Anomaly score {score:.4g} < acceptance threshold {th.pass_threshold:.4g}"],
                    "none", norm, False)
