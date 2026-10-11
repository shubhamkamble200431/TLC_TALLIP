"""Unit tests 11, 12, 13, 14: aggregation, bootstrap, factorial design, heatmap."""
import numpy as np
import pytest

from src.statistics.aggregate import _ci95, aggregate_by_config
from src.statistics.bootstrap import holm_bonferroni, paired_bootstrap
from src.statistics.factorial import build_design, design_row, fit, term_names
from src.config import ALL_CASE_IDS


# ---- aggregation (section 14) ----------------------------------
def test_ci95_basic():
    mean, sd, lo, hi, n = _ci95([10.0, 12.0, 14.0])
    assert mean == 12.0 and n == 3
    assert lo < mean < hi
    assert _ci95([])[4] == 0
    assert _ci95([5.0])[1] == 0.0


def test_aggregate_by_config_shapes():
    recs = []
    for seed, off in zip((42, 43, 44), (0.0, 0.5, -0.5)):
        recs.append({
            "case_id": "X-0000", "seed": seed, "status": "completed", "bucket_rows": [],
            "metrics": {"bleu": 30 + off, "comet": 60 + off, "chrf2": 50 + off,
                        "ter": 40, "meteor": 45, "bertscore": 70, "length_ratio": 1.0,
                        "absolute_length_error": 2.0, "relative_length_error": 0.1,
                        "premature_eos_rate": 0.0, "overgeneration_rate": 0.0, "truncation_rate": 0.0},
        })
    rows = aggregate_by_config(recs)
    assert len(rows) == 1
    r = rows[0]
    assert r["config"] == "X-0000" and r["seed_count"] == 3
    assert abs(r["bleu_mean"] - 30.0) < 1e-9
    assert r["bleu_ci_low"] < r["bleu_mean"] < r["bleu_ci_high"]


# ---- bootstrap (section 17) -----------------------------------
def test_paired_bootstrap_identical_systems():
    refs = [f"the cat sat number {i}" for i in range(60)]
    res = paired_bootstrap(refs, list(refs), list(refs), metric="bleu", n_samples=300, seed=1)
    assert abs(res.observed_diff) < 1e-6
    assert res.p_value > 0.2
    assert res.ci_low <= 0 <= res.ci_high


def test_paired_bootstrap_precomputed_detects_difference():
    a = [1.0] * 100
    b = [0.0] * 100
    res = paired_bootstrap(["x"] * 100, ["x"] * 100, ["x"] * 100, metric="precomputed",
                           seg_scores_a=a, seg_scores_b=b, n_samples=500, seed=2)
    assert res.observed_diff == pytest.approx(1.0)
    assert res.p_value < 0.05


def test_holm_bonferroni_monotone_and_corrects_upward():
    class R:
        def __init__(self, p):
            self.p_value = p
            self.p_value_corrected = None
            self.reject_null = None

    rs = [R(0.001), R(0.02), R(0.04), R(0.5)]
    holm_bonferroni(rs, alpha=0.05)
    cps = [r.p_value_corrected for r in rs]
    assert all(cps[i] <= cps[i + 1] + 1e-12 for i in range(len(cps) - 1))
    assert cps[0] >= 0.001 * 4 - 1e-12


# ---- factorial design (section 18) ---------------------------
def test_term_names_count_and_content():
    names = term_names()
    assert len(names) == 16
    assert names[0] == "Intercept"
    assert "ABCD" in names and "AB" in names


def test_design_is_orthogonal_over_all_16():
    X, _ = build_design(ALL_CASE_IDS)
    gram = X.T @ X
    # +/-1 orthogonal coding over the full factorial -> X^T X = 16 * I
    assert np.allclose(gram, 16 * np.eye(16))


def test_fit_recovers_known_effects():
    # y = 10 + 2*A + (-1)*B + 0.5*AB  (A,B in {-1,+1})
    from src.config import parse_case_id

    data = {}
    for cid in ALL_CASE_IDS:
        f = parse_case_id(cid)
        A = 1 if f["bbs"] else -1
        B = 1 if f["tlc"] else -1
        y = 10 + 2 * A - 1 * B + 0.5 * A * B
        data[cid] = [y, y + 0.01, y - 0.01]  # 3 "seeds"
    effects = {e.term: e for e in fit(data)}
    assert effects["A"].coefficient == pytest.approx(2.0, abs=0.05)
    assert effects["B"].coefficient == pytest.approx(-1.0, abs=0.05)
    assert effects["AB"].coefficient == pytest.approx(0.5, abs=0.05)
    assert effects["C"].coefficient == pytest.approx(0.0, abs=0.05)
    assert effects["A"].significant
    assert not effects["C"].significant


def test_fit_requires_all_16():
    with pytest.raises(ValueError):
        fit({"X-0000": [1.0], "X-1111": [2.0]})


# ---- heatmap (section 12) -----------------------------------
def test_annotated_heatmap_writes_files(tmp_path):
    from src.visualization.heatmaps import annotated_heatmap

    m = np.arange(12, dtype=float).reshape(3, 4)
    out = tmp_path / "hm"
    paths = annotated_heatmap(m, ["r1", "r2", "r3"], ["c1", "c2", "c3", "c4"], str(out),
                              title="t", cbar_label="v", cmap="viridis")
    assert any(p.endswith(".png") for p in paths)
    assert any(p.endswith(".pdf") for p in paths)
    for p in paths:
        import os

        assert os.path.getsize(p) > 0
