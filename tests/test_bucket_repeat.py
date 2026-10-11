"""
Bucket-cap x repetition INFERENCE ablation (reviewer request):
sampling determinism / fairness + the repetition-invariance claim.
"""
import importlib.util
import os
import sys

import pytest

_SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name.replace(".py", ""), os.path.join(_SCRIPTS, name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bra = _load("run_bucket_repeat_ablation.py")


# ---- deterministic_order --------------------------------------------------
def test_deterministic_order_is_a_permutation():
    o = bra.deterministic_order(50, 42)
    assert sorted(o) == list(range(50))


def test_deterministic_order_reproducible_and_seed_sensitive():
    assert bra.deterministic_order(100, 42) == bra.deterministic_order(100, 42)
    assert bra.deterministic_order(100, 42) != bra.deterministic_order(100, 43)


# ---- pick_indices: duplicate mode --------------------------------------
def test_duplicate_mode_same_prefix_every_copy():
    order_b = bra.deterministic_order(200, 42 + 5)
    k = 36
    picks = [bra.pick_indices("duplicate", order_b, 200, k, j, 42, 5) for j in range(10)]
    # every repetition draws the identical k-prefix -> selection is repetition-invariant
    assert all(p == order_b[:k] for p in picks)
    assert len(picks[0]) == k


def test_duplicate_mode_nested_prefixes():
    order_b = bra.deterministic_order(200, 7)
    small = bra.pick_indices("duplicate", order_b, 200, 20, 0, 42, 2)
    big = bra.pick_indices("duplicate", order_b, 200, 60, 0, 42, 2)
    assert big[:20] == small          # 0.2M subset of 0.6M subset ... (nested)


def test_duplicate_mode_slice_caps_at_pool_size():
    order_b = bra.deterministic_order(30, 1)
    picks = bra.pick_indices("duplicate", order_b, 30, 100, 0, 42, 1)
    assert len(picks) == 30 and sorted(picks) == list(range(30))


# ---- pick_indices: resample mode -------------------------------------
def test_resample_mode_returns_exactly_k_and_in_range():
    picks = bra.pick_indices("resample", [], 178, 178, 3, 42, 5)
    assert len(picks) == 178
    assert all(0 <= i < 178 for i in picks)


def test_resample_mode_varies_across_copies_but_is_deterministic():
    a0 = bra.pick_indices("resample", [], 178, 100, 0, 42, 5)
    a0b = bra.pick_indices("resample", [], 178, 100, 0, 42, 5)
    a1 = bra.pick_indices("resample", [], 178, 100, 1, 42, 5)
    assert a0 == a0b          # same (j, seed, bucket) -> same draw
    assert a0 != a1           # different copy index -> different draw


def test_resample_mode_wraps_when_k_exceeds_pool():
    picks = bra.pick_indices("resample", [], 10, 25, 0, 42, 3)
    assert len(picks) == 25
    assert set(picks) == set(range(10))     # every pool item used, with wrap


# ---- the repetition-invariance claim the runner prints ------------------
def test_corpus_bleu_is_invariant_to_uniform_repetition():
    from src.evaluation.metrics import corpus_bleu

    preds = ["राम घर जाता है", "यह एक परीक्षण है", "वह विद्यालय गया"]
    refs = ["राम घर जाता है", "यह एक छोटा परीक्षण है", "वह विद्यालय गया था"]
    base = corpus_bleu(preds, refs)
    for r in (2, 3, 5, 10):
        assert corpus_bleu(preds * r, refs * r) == pytest.approx(base, abs=1e-9)


# ---- M (per-bucket cap) semantics -------------------------------------
def test_M_is_min_bucket_count():
    counts = {1: 4157, 2: 1743, 3: 871, 4: 315, 5: 178}
    assert min(counts.values()) == 178          # v2 unsampled -> B5
    counts_v1 = {1: 3945, 2: 975, 3: 345, 4: 93, 5: 56}
    assert min(counts_v1.values()) == 56        # v1 unsampled -> B5
