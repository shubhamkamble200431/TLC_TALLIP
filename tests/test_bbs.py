"""Unit tests 3, 4, 5, 6: BBS normalisation, monotonicity, candidate generation, sampler."""
import numpy as np
import pytest

from src.data.dataset import SaHiDataset, build_weighted_sampler, normalized_bucket_weights
from src.experiments.bbs_grid import SearchSpace, is_monotonic, normalize_mean


# ---- normalisation (section 6) ------------------------------------
def test_normalize_mean_sums_to_five():
    for vec in ([1, 1, 1, 1, 1], [0.5, 0.75, 1.0, 1.5, 2.0], [2, 2, 2, 2, 2]):
        n = normalize_mean(vec)
        assert abs(sum(n) - 5.0) < 1e-9
        assert abs(np.mean(n) - 1.0) < 1e-9


def test_normalize_mean_preserves_ratios():
    n = normalize_mean([1, 2, 3, 4, 5])
    r = [x / n[0] for x in n]
    assert np.allclose(r, [1, 2, 3, 4, 5])


def test_normalize_rejects_nonpositive():
    with pytest.raises(ValueError):
        normalize_mean([0, 1, 1, 1, 1])


def test_normalized_bucket_weights_matches():
    assert np.allclose(normalized_bucket_weights([1, 1, 1.2, 1.5, 2.0]),
                       normalize_mean([1, 1, 1.2, 1.5, 2.0]))


# ---- monotonicity (section 7) -----------------------------------
def test_is_monotonic():
    assert is_monotonic([1, 1, 1.2, 1.5, 2.0])
    assert not is_monotonic([1, 1, 0.9, 1.5, 2.0])


def test_monotonic_filter_in_space():
    sp = SearchSpace(values=[0.5, 1.0, 1.5, 2.0], mode="grid2d", coordinate_axes=[3, 4],
                     monotonic=True, max_candidates=999)
    cands = sp.generate()
    for c in cands:
        assert is_monotonic(c.normalized)


# ---- candidate generation (section 7) --------------------------
def test_dedup_after_normalisation():
    sp = SearchSpace(values=[1.0, 2.0], mode="exhaustive", monotonic=False, max_candidates=999)
    cands = sp.generate()
    keys = {tuple(round(x, 4) for x in c.normalized) for c in cands}
    assert len(keys) == len(cands)  # no duplicates


def test_always_include_survives_cap():
    sp = SearchSpace(values=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0], mode="grid2d",
                     coordinate_axes=[3, 4], monotonic=True, max_candidates=2,
                     always_include={"manual": [1.0, 1.0, 1.2, 1.5, 2.0],
                                     "uniform": [1, 1, 1, 1, 1]})
    cands = sp.generate()
    tags = {c.tag for c in cands}
    assert "manual" in tags and "uniform" in tags
    assert len(cands) <= 2 + 2


def test_grid2d_only_touches_two_axes():
    anchor = [1.0, 1.0, 1.0, 1.0, 1.0]
    sp = SearchSpace(values=[0.5, 1.0, 2.0], mode="grid2d", coordinate_axes=[3, 4],
                     anchor=anchor, monotonic=False, normalize_mean=False, max_candidates=999)
    for c in sp.generate():
        assert c.raw[0] == c.raw[1] == c.raw[2] == 1.0


# ---- sampler determinism + emphasis (section 4) ----------------
class _Tok:
    pad_token_id = 0

    def __call__(self, *a, **k):
        return {"input_ids": [1, 2], "attention_mask": [1, 1]}

    def get_vocab(self):
        return {}


def _toy_dataset():
    # 100 B1 (short) + 100 B5 (long) source sentences
    src = ["w"] * 100 + [" ".join(["w"] * 30)] * 100
    tgt = ["x"] * 200
    return SaHiDataset(src, tgt, _Tok(), model_family="nllb",
                      max_input_length=8, max_target_length=8)


def test_weighted_sampler_is_deterministic_given_seed():
    ds = _toy_dataset()
    s1 = list(build_weighted_sampler(ds, [1, 1, 1, 1, 3.0], seed=42))
    s2 = list(build_weighted_sampler(ds, [1, 1, 1, 1, 3.0], seed=42))
    s3 = list(build_weighted_sampler(ds, [1, 1, 1, 1, 3.0], seed=43))
    assert s1 == s2
    assert s1 != s3


def test_weighted_sampler_upweights_high_buckets():
    ds = _toy_dataset()  # idx 0..99 -> B1, idx 100..199 -> B5
    idxs = list(build_weighted_sampler(ds, [1, 1, 1, 1, 4.0], seed=42))
    frac_b5 = np.mean([i >= 100 for i in idxs])
    # B5 weight 4x after mean-normalisation -> should dominate draws
    assert frac_b5 > 0.6


def test_stratified_subsample_preserves_bucket_proportions():
    import numpy as np
    from src.data.dataset import stratified_subsample
    from src.data.buckets import assign_bucket, n_tokens
    rng = np.random.default_rng(0)
    # synthetic corpus: bucket mix 60/20/12/5/3
    lens = ([3] * 6000 + [8] * 2000 + [13] * 1200 + [18] * 500 + [30] * 300)
    rng.shuffle(lens)
    src = [" ".join(["w"] * n) for n in lens]
    tgt = [f"t{i}" for i in range(len(src))]
    B = [5, 10, 15, 20]

    def pct(s):
        c = [0] * 5
        for x in s:
            c[assign_bucket(n_tokens(x), B) - 1] += 1
        return [v / len(s) for v in c]

    full = pct(src)
    for N in (1000, 4000):
        ss, st = stratified_subsample(src, tgt, N, B, seed=42)
        assert len(ss) == len(st) == N
        assert max(abs(a - b) for a, b in zip(pct(ss), full)) < 0.01   # within 1 pp
    # deterministic
    a = stratified_subsample(src, tgt, 2000, B, seed=7)
    b = stratified_subsample(src, tgt, 2000, B, seed=7)
    assert a == b
