"""Unit tests for the ML core: decoding safety, decision policy, metrics, thresholds, splits, degradations."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image
from sklearn.metrics import roc_auc_score

from defectml.data.manifest import Sample
from defectml.data.splits import SplitConfig, leakage_report, make_splits
from defectml.decision import FAIL, PASS, REVIEW, PolicyConfig, Thresholds, decide
from defectml.evaluation import degradations as D
from defectml.evaluation.metrics import (
    bootstrap_auroc_ci,
    calibration_metrics,
    platt_apply,
    platt_fit,
    ranking_metrics,
    select_threshold_f1,
    select_threshold_youden,
    threshold_for_recall,
    wilson_interval,
)
from defectml.imaging import ImageDecodeError, assess_quality, decode_image_bytes, sha256_bytes
from defectml.localization import extract_regions
from defectml.models.patchcore import greedy_coreset
from defectml.training import choose_thresholds


def _png(arr: np.ndarray, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return buf.getvalue()


# --------------------------------------------------------------------------- image decoding
class TestDecoding:
    def test_decodes_rgb_png(self):
        img = np.random.default_rng(0).integers(0, 255, (80, 96, 3), dtype=np.uint8)
        out = decode_image_bytes(_png(img))
        assert out.shape == (80, 96, 3) and out.dtype == np.uint8
        assert np.array_equal(out, img)

    def test_grayscale_becomes_rgb(self):
        out = decode_image_bytes(_png(np.full((70, 70), 128, np.uint8)))
        assert out.shape == (70, 70, 3)

    @pytest.mark.parametrize("data", [b"", b"not an image at all", b"\x89PNG\r\n\x1a\n" + b"\x00" * 40])
    def test_rejects_garbage(self, data):
        with pytest.raises(ImageDecodeError):
            decode_image_bytes(data)

    def test_rejects_truncated_file(self):
        data = _png(np.random.default_rng(1).integers(0, 255, (128, 128, 3), dtype=np.uint8))
        with pytest.raises(ImageDecodeError):
            decode_image_bytes(data[: len(data) // 2])

    def test_rejects_too_many_pixels_before_decoding(self):
        # header claims a large image; must be rejected from the header alone
        data = _png(np.zeros((300, 300, 3), np.uint8))
        with pytest.raises(ImageDecodeError, match="too large"):
            decode_image_bytes(data, max_pixels=300 * 299)

    def test_rejects_tiny_images(self):
        with pytest.raises(ImageDecodeError, match="too small"):
            decode_image_bytes(_png(np.zeros((8, 8, 3), np.uint8)))

    def test_rejects_disallowed_format(self):
        buf = io.BytesIO()
        Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(buf, format="GIF")
        with pytest.raises(ImageDecodeError, match="Unsupported"):
            decode_image_bytes(buf.getvalue())

    def test_sha256_is_stable(self):
        assert sha256_bytes(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

    def test_quality_flags_dark_and_blurry(self):
        dark = np.full((128, 128, 3), 5, np.uint8)
        q = assess_quality(dark)
        assert not q.ok and q.issues


# --------------------------------------------------------------------------- decision policy
TH = Thresholds(fail_threshold=2.0, pass_threshold=1.5, ood_threshold=10.0)


class TestDecisionPolicy:
    def test_fail_at_or_above_reject_threshold(self):
        assert decide(2.0, TH).decision == FAIL
        assert decide(3.5, TH).decision == FAIL

    def test_pass_only_below_pass_threshold_without_flags(self):
        d = decide(1.0, TH)
        assert d.decision == PASS and not d.needs_review

    def test_borderline_goes_to_review(self):
        d = decide(1.7, TH)
        assert d.decision == REVIEW and d.needs_review

    @pytest.mark.parametrize("kw", [{"error": "decoder crashed"}, {"error": None}])
    def test_inspection_impossible_never_passes(self, kw):
        assert decide(None, TH, **kw).decision == REVIEW
        assert decide(1.0, None).decision == REVIEW

    def test_nan_score_is_review(self):
        assert decide(float("nan"), TH).decision == REVIEW

    def test_quality_issue_blocks_auto_pass(self):
        assert decide(0.5, TH, quality_issues=["too dark"]).decision == REVIEW

    def test_ood_blocks_auto_pass(self):
        assert decide(0.5, TH, ood_score=50.0).decision == REVIEW
        assert decide(0.5, TH, ood_score=5.0).decision == PASS

    def test_secondary_model_disagreement(self):
        assert decide(0.5, TH, secondary_decision=FAIL).decision == REVIEW
        d = decide(2.5, TH, secondary_decision=PASS)
        assert d.decision == FAIL and d.needs_review

    def test_policy_switches_are_respected(self):
        cfg = PolicyConfig(review_on_quality_issues=False, review_on_ood=False)
        assert decide(0.5, TH, quality_issues=["dark"], ood_score=99.0, cfg=cfg).decision == PASS

    def test_critical_fail_is_queued_for_confirmation(self):
        d = decide(10.0, TH)  # ratio 5 >= critical_ratio 2.5
        assert d.decision == FAIL and d.severity == "critical" and d.needs_review

    def test_thresholds_validate(self):
        with pytest.raises(ValueError):
            Thresholds(1.0, 2.0).validate()


# --------------------------------------------------------------------------- metrics & thresholds
class TestMetrics:
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(60), np.ones(40)].astype(int)
    s = np.r_[rng.normal(0, 1, 60), rng.normal(1.5, 1, 40)]

    def test_auroc_matches_sklearn(self):
        assert ranking_metrics(self.y, self.s)["auroc"] == pytest.approx(roc_auc_score(self.y, self.s))

    def test_bootstrap_ci_contains_point_estimate(self):
        lo, hi = bootstrap_auroc_ci(self.y, self.s, n_boot=200)
        assert lo <= roc_auc_score(self.y, self.s) <= hi

    def test_wilson_interval(self):
        lo, hi = wilson_interval(9, 10)
        assert 0.5 < lo < 0.9 < hi <= 1.0
        assert wilson_interval(0, 0) is None

    def test_threshold_for_recall_achieves_target(self):
        t = threshold_for_recall(self.y, self.s, 0.95)
        assert (self.s[self.y == 1] >= t).mean() >= 0.95

    def test_youden_is_prevalence_independent_unlike_f1(self):
        """Replicating the anomalies 5x must not move the Youden threshold, but drags F1 down (the bug we fixed)."""
        rng = np.random.default_rng(3)
        n, a = rng.normal(0, 1, 30), rng.normal(2, 1, 30)
        y1, s1 = np.r_[np.zeros(30), np.ones(30)], np.r_[n, a]
        y5, s5 = np.r_[np.zeros(30), np.ones(150)], np.r_[n, np.tile(a, 5)]
        assert select_threshold_youden(y1, s1) == pytest.approx(select_threshold_youden(y5, s5))
        assert select_threshold_f1(y5, s5) <= select_threshold_f1(y1, s1)

    def test_choose_thresholds_orders_pass_below_fail_with_review_band(self):
        th = choose_thresholds(self.y, self.s, target_recall=0.98)
        assert th["pass_threshold"] < th["fail_threshold"]
        assert "Youden" in th["method"]

    def test_choose_thresholds_keeps_review_band_when_separable(self):
        y = np.r_[np.zeros(20), np.ones(20)]
        s = np.r_[np.linspace(0, 1, 20), np.linspace(5, 6, 20)]
        th = choose_thresholds(y, s)
        assert th["pass_threshold"] < th["fail_threshold"]

    def test_platt_calibration_is_monotone_and_bounded(self):
        cal = platt_fit(self.y, self.s)
        p = platt_apply(cal, np.linspace(-3, 5, 50))
        assert np.all(np.diff(p) >= 0) and p.min() >= 0 and p.max() <= 1
        cm = calibration_metrics(self.y, platt_apply(cal, self.s))
        assert 0 <= cm["ece"] <= 1 and 0 <= cm["brier"] <= 1


# --------------------------------------------------------------------------- splits & leakage
def _samples(n_groups=12, per=5, n_anom=20):
    out = [Sample(path=f"n/{g}_{i}.png", label="normal", group=f"g{g}") for g in range(n_groups) for i in range(per)]
    out += [Sample(path=f"a/{i}.png", label="anomaly", defect_type="scratch", group=f"ga{i % 6}") for i in range(n_anom)]
    return out


class TestSplits:
    def test_train_is_normal_only_and_all_splits_present(self):
        s = make_splits(_samples())
        assert {x.split for x in s} == {"train", "val", "test"}
        assert all(not x.is_anomaly for x in s if x.split == "train")

    def test_groups_never_straddle_splits(self):
        s = make_splits(_samples())
        by_group: dict[str, set] = {}
        for x in s:
            by_group.setdefault(x.group, set()).add(x.split)
        assert all(len(v) == 1 for v in by_group.values())
        assert leakage_report(s)["leakage_free"]

    def test_deterministic_for_seed(self):
        a = [x.split for x in make_splits(_samples(), SplitConfig(seed=1))]
        b = [x.split for x in make_splits(_samples(), SplitConfig(seed=1))]
        assert a == b

    def test_detects_exact_duplicates_across_splits(self):
        s = make_splits(_samples())
        tr = next(i for i, x in enumerate(s) if x.split == "train")
        te = next(i for i, x in enumerate(s) if x.split == "test")
        sha = [f"h{i}" for i in range(len(s))]
        sha[te] = sha[tr]
        rep = leakage_report(s, sha=sha)
        assert not rep["leakage_free"] and rep["exact_duplicates_across_splits"]


# --------------------------------------------------------------------------- degradations
_yy, _xx = np.mgrid[0:96, 0:128]
# smooth, structured "part" image (gradients + a disc) - pure noise would make geometric shifts saturate instantly
IMG = np.stack([40 + _xx * 1.4, 60 + _yy * 1.6, 120 + 60 * np.sin(_xx / 18.0) * np.cos(_yy / 14.0)], -1)
IMG[(_yy - 48) ** 2 + (_xx - 64) ** 2 < 22 ** 2] = (210, 190, 60)
IMG = np.clip(IMG, 0, 255).astype(np.uint8)
MASK = np.zeros((96, 128), np.uint8)
MASK[30:50, 40:70] = 1


@pytest.mark.parametrize("name", sorted(D.DEGRADATIONS))
def test_degradation_contract(name):
    prev = None
    for sev in (1, 3, 5):
        img, mask = D.apply(name, IMG, MASK, sev, seed=0)
        assert img.shape == IMG.shape and img.dtype == np.uint8
        assert mask is None or mask.shape == MASK.shape
        again, _ = D.apply(name, IMG, MASK, sev, seed=0)
        assert np.array_equal(img, again), "degradations must be deterministic for a seed"
        diff = float(np.abs(img.astype(int) - IMG.astype(int)).mean())
        if prev is not None and name not in ("orientation", "background_change"):
            assert diff >= prev - 1.0, f"{name}: severity 5 should not be milder than severity 1"
        prev = diff
    with pytest.raises(ValueError):
        D.apply("no-such-degradation", IMG, MASK, 1)


# --------------------------------------------------------------------------- localisation & coreset
def test_regions_are_in_original_image_coordinates():
    amap = np.zeros((32, 32), np.float32)
    amap[8:16, 16:24] = 5.0
    regions, mask = extract_regions(amap, 1.0, (320, 640))
    assert len(regions) == 1 and mask.shape == (320, 640)
    x, y, w, h = regions[0]["bbox"]
    assert 300 <= x <= 340 and 70 <= y <= 90 and 140 <= w <= 180 and 70 <= h <= 90
    assert extract_regions(np.zeros((32, 32), np.float32), 1.0, (320, 640))[0] == []


def test_greedy_coreset_matches_naive_k_center():
    """The memory-lean distance update must select exactly what the textbook k-center loop selects."""
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(400, 16)).astype(np.float32)
    got = greedy_coreset(feats, 25, proj_dim=128, seed=3)  # dim < proj_dim -> no projection
    r = np.random.default_rng(3)
    idx = int(r.integers(0, len(feats)))
    sel, min_d = [], np.full(len(feats), np.inf)
    for _ in range(25):
        sel.append(idx)
        min_d = np.minimum(min_d, ((feats - feats[idx]) ** 2).sum(1))
        idx = int(np.argmax(min_d))
    assert list(got) == sel
    assert len(set(got.tolist())) == 25


def test_fail_without_segmented_region_gets_approximate_peak_box():
    """A FAIL whose map never reaches the (pixel-F1) segmentation threshold must still show the operator where to look."""
    from defectml.inference import Inspector
    from defectml.models.base import Prediction

    amap = np.full((32, 32), 0.5, np.float32)
    amap[10:16, 20:26] = 2.0  # image score 2.0: above fail (1.5), below seg threshold (3.0)

    class _Model:
        config = {"input_size": 32}

        def predict_batch(self, imgs):
            return [Prediction(score=float(amap.max()), anomaly_map=amap.copy(), ood_score=0.0)]

    insp = Inspector.__new__(Inspector)
    insp.model, insp.classifier = _Model(), None
    insp.meta = {"name": "fake", "algorithm": "fake",
                 "thresholds": {"fail_threshold": 1.5, "pass_threshold": 1.0, "seg_threshold": 3.0, "ood_threshold": None}}
    img = np.full((128, 128, 3), 128, np.uint8)
    img[::8] = 90  # some texture so quality checks pass
    out = insp.inspect(img, render=False)
    assert out.decision.decision == "FAIL"
    assert len(out.regions) == 1 and out.regions[0]["approximate"] is True
    x, y, w, h = out.regions[0]["bbox"]  # map cell (row 10-16, col 20-26) of 32 -> image 4x scale
    assert 70 <= x <= 84 and 30 <= y <= 44 and w <= 40 and h <= 40


def test_cli_uses_app_storage_dirs_and_resolves_bare_names(tmp_path, monkeypatch):
    """`defectml` must read/write the same directories as the web app, and accept registered names."""
    import argparse

    from defectml import cli

    ds_dir, models_dir = tmp_path / "datasets", tmp_path / "models"
    (ds_dir / "visa-pcb1").mkdir(parents=True)
    (models_dir / "visa-pcb1-patchcore").mkdir(parents=True)
    monkeypatch.setattr(cli, "_storage_dirs", lambda: (ds_dir, models_dir))

    a = argparse.Namespace(cmd="train", dataset="visa-pcb1", algorithm="padim", out=None)
    cli._normalise_args(a)
    assert a.dataset == str(ds_dir / "visa-pcb1") and a.out == str(models_dir / "visa-pcb1-padim")

    a = argparse.Namespace(cmd="evaluate", dataset="visa-pcb1", models=["visa-pcb1-patchcore", "nope"], out=None)
    cli._normalise_args(a)
    assert a.models == [str(models_dir / "visa-pcb1-patchcore"), "nope"]  # unknown names are left untouched

    a = argparse.Namespace(cmd="prepare-visa", out=None)
    cli._normalise_args(a)
    assert a.out == str(ds_dir)

    explicit = tmp_path / "elsewhere"
    a = argparse.Namespace(cmd="import-folder", out=str(explicit), src="x")
    cli._normalise_args(a)
    assert a.out == str(explicit)  # explicit paths always win
