"""V1 + alpha*V2 data-scale ablation: nesting, determinism, repetition, heatmap."""
import os

import numpy as np
import pytest

from src.data.v1v2 import compose, nested_v2_subset


V2 = ([f"s{i}" for i in range(1000)], [f"t{i}" for i in range(1000)])


def test_subset_sizes_round_alpha():
    for a in (0.0, 0.1, 0.25, 0.5, 1.0):
        s, t = nested_v2_subset(*V2, alpha=a, subset_seed=7)
        assert len(s) == len(t) == round(a * 1000)


def test_subsets_are_nested_across_alpha():
    picks = {}
    for a in (0.1, 0.2, 0.4, 0.8, 1.0):
        s, _ = nested_v2_subset(*V2, alpha=a, subset_seed=7)
        picks[a] = set(s)
    alphas = sorted(picks)
    for lo, hi in zip(alphas, alphas[1:]):
        assert picks[lo].issubset(picks[hi]), f"{lo} not subset of {hi}"


def test_subset_is_deterministic_given_seed():
    a1 = nested_v2_subset(*V2, alpha=0.37, subset_seed=42)
    a2 = nested_v2_subset(*V2, alpha=0.37, subset_seed=42)
    a3 = nested_v2_subset(*V2, alpha=0.37, subset_seed=43)
    assert a1 == a2
    assert a1 != a3


def _fake_corpora(tmp_path, n_v1=20, n_v2=30):
    v1 = tmp_path / "v1"
    v2 = tmp_path / "v2"
    v1.mkdir()
    v2.mkdir()
    (v1 / "train.sa").write_text("\n".join(f"v1src {i}" for i in range(n_v1)) + "\n")
    (v1 / "train.hi").write_text("\n".join(f"v1tgt {i}" for i in range(n_v1)) + "\n")
    (v2 / "train.sa").write_text("\n".join(f"v2src {i}" for i in range(n_v2)) + "\n")
    (v2 / "train.hi").write_text("\n".join(f"v2tgt {i}" for i in range(n_v2)) + "\n")
    return str(v1), str(v2)


def test_alpha_zero_is_v1_only_and_alpha_one_is_full(tmp_path):
    v1, v2 = _fake_corpora(tmp_path, n_v1=20, n_v2=30)

    s0, _, m0 = compose(0.0, 1, v1, v2, subset_seed=1)
    assert m0.n_v2_used == 0
    assert len(s0) == m0.n_v1 == 20 and m0.n_v2_pool == 30

    s1, _, m1 = compose(1.0, 1, v1, v2, subset_seed=1)
    assert m1.n_v2_used == m1.n_v2_pool == 30
    assert m1.n_composed == 20 + 30 == len(s1)


def test_repetitions_scale_training_examples(tmp_path):
    v1, v2 = _fake_corpora(tmp_path, n_v1=30, n_v2=40)
    _, _, m1 = compose(0.5, 1, v1, v2, subset_seed=1)
    _, _, m3 = compose(0.5, 3, v1, v2, subset_seed=1)
    assert m3.training_examples == 3 * m1.n_composed
    assert m1.training_examples == m1.n_composed
    assert m1.n_composed == 30 + round(0.5 * 40)


def test_train_repeat_factor_replicates_list():
    # the trainer applies `train_src * k`; assert that idiom is order-preserving
    base = ["a", "b", "c"]
    k = 3
    rep = base * k
    assert len(rep) == 9 and rep[:3] == base and rep[3:6] == base


def test_heatmap_from_csv(tmp_path):
    import sys
    import pandas as pd
    import importlib.util

    scripts_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, os.path.abspath(scripts_dir))

    csv = tmp_path / "v1_v2_scale_ablation.csv"
    rows = []
    for rep in (1, 2, 3):
        for a in [round(i / 10, 1) for i in range(11)]:
            rows.append({"repetitions": rep, "v2_fraction": f"{a:.1f}",
                         "training_examples": 58106 + int(a * 15921) * rep,
                         "bleu": 30 + a * 4 + rep * 0.5, "chrf2": 50 + a * 3,
                         "comet": 60 + a * 5, "length_error": 2.5 - a * 0.3,
                         "seed": 42, "status": "done"})
    pd.DataFrame(rows).to_csv(csv, index=False)

    spec = importlib.util.spec_from_file_location(
        "gen_v1v2", os.path.join(os.path.dirname(__file__), "..", "scripts", "generate_v1_v2_heatmaps.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    df = pd.read_csv(csv, dtype={"v2_fraction": str})
    mat, rlab, clab = mod._pivot(df, "bleu")
    assert mat.shape == (3, 11)
    assert clab[0] == "V1+0.0V2" and clab[-1] == "V1+1.0V2"
    assert rlab == ["1 rep", "2 reps", "3 reps"]
    assert np.isfinite(mat).all()
