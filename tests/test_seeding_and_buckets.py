"""Unit tests 2, 10: seed reproducibility + length-bucket assignment."""
import random

import numpy as np

from src.data.buckets import assign_bucket, bucket_counts, DEFAULT_BOUNDARIES
from src.utils.seeding import make_generator, set_global_seed


def _draw():
    import torch

    return (
        random.random(),
        float(np.random.rand()),
        float(torch.rand(1).item()),
    )


def test_set_global_seed_is_reproducible():
    set_global_seed(42, deterministic=False)
    a = _draw()
    set_global_seed(42, deterministic=False)
    b = _draw()
    assert a == b

    set_global_seed(43, deterministic=False)
    c = _draw()
    assert c != a


def test_make_generator_deterministic():
    import torch

    g1 = make_generator(42)
    g2 = make_generator(42)
    assert torch.equal(torch.randperm(50, generator=g1), torch.randperm(50, generator=g2))


def test_seed_report_fields():
    rep = set_global_seed(44, deterministic=True)
    d = rep.to_dict()
    assert d["seed"] == 44
    assert set(("python_hashseed", "cudnn_deterministic", "nondeterministic_ops")).issubset(d)


def test_bucket_boundaries():
    assert DEFAULT_BOUNDARIES == [5, 10, 15, 20]
    assert assign_bucket(1) == 1
    assert assign_bucket(5) == 1
    assert assign_bucket(6) == 2
    assert assign_bucket(15) == 3
    assert assign_bucket(20) == 4
    assert assign_bucket(21) == 5
    assert assign_bucket(999) == 5


def test_bucket_counts_natural_membership():
    sents = ["a b c", "a " * 8, "w " * 12, "w " * 18, "w " * 40]
    counts = bucket_counts([s.strip() for s in sents])
    assert counts == {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}
