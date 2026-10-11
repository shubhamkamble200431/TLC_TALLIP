"""
V1 + alpha*V2 training-set construction for the data-scale ablation.

Two REAL Sanskrit->Hindi corpora, used exactly as pasted into `data/` (no
synthetic augmentation):

    V1 : data/v1/train.{sa,hi}   (the older / shorter corpus)   -- kept FIXED
    V2 : data/v2/train.{sa,hi}   (the main corpus, also used by every headline
                                  experiment)

    V1 + alpha*V2  =  V1  ++  V2_subset(alpha)

`V2_subset(alpha)` is a NESTED / subset-consistent selection: ONE deterministic
permutation of the V2 training pairs (seeded by `subset_seed`) is taken, and
alpha selects the prefix of length round(alpha * |V2_train|).  Hence
    V2_subset(0.1) subset V2_subset(0.2) subset ... subset V2_subset(1.0)
so larger alpha strictly *adds* V2 examples rather than re-sampling.

No de-duplication against V1 is performed (per the "use the data exactly as
pasted" decision); at alpha=1.0 the composed set is simply
|V1_train| + |V2_train| pairs, with any V1/V2 overlap present twice.  The
composed list is then replicated `repetitions` times by `train_repeat_factor`
in the trainer, so `training_examples = |V1 + alpha*V2| * repetitions`.

Evaluation for this ablation uses the SAME held-out set as the main
experiments (V2 val + V2 test_7264_5) -- the runner resolves it via
dataset_version="v2".
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import List, Tuple

from src.utils.io import read_lines


@dataclass
class Composition:
    alpha: float
    repetitions: int
    n_v1: int
    n_v2_pool: int
    n_v2_used: int
    n_composed: int              # |V1 + alpha*V2|
    training_examples: int       # n_composed * repetitions
    subset_seed: int

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _load_pairs(data_dir: str, src_name: str, tgt_name: str) -> Tuple[List[str], List[str]]:
    src = read_lines(os.path.join(data_dir, src_name))
    tgt = read_lines(os.path.join(data_dir, tgt_name))
    if len(src) != len(tgt):
        raise ValueError(f"{data_dir}: src/tgt mismatch ({len(src)} vs {len(tgt)})")
    return src, tgt


def load_v1(v1_dir: str = "data/v1", src_name: str = "train.sa", tgt_name: str = "train.hi"):
    return _load_pairs(v1_dir, src_name, tgt_name)


def load_v2(v2_dir: str = "data/v2", src_name: str = "train.sa", tgt_name: str = "train.hi"):
    return _load_pairs(v2_dir, src_name, tgt_name)


def nested_v2_subset(
    v2_src: List[str], v2_tgt: List[str], alpha: float, subset_seed: int
) -> Tuple[List[str], List[str]]:
    """Prefix of a single deterministic permutation -> nested across alpha."""
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    order = list(range(len(v2_src)))
    random.Random(int(subset_seed)).shuffle(order)
    k = round(alpha * len(v2_src))
    picked = order[:k]
    return [v2_src[i] for i in picked], [v2_tgt[i] for i in picked]


def compose(
    alpha: float,
    repetitions: int,
    v1_dir: str = "data/v1",
    v2_dir: str = "data/v2",
    subset_seed: int = 42,
    v1_src_name: str = "train.sa",
    v1_tgt_name: str = "train.hi",
    v2_src_name: str = "train.sa",
    v2_tgt_name: str = "train.hi",
) -> Tuple[List[str], List[str], Composition]:
    v1_src, v1_tgt = load_v1(v1_dir, v1_src_name, v1_tgt_name)
    v2_src, v2_tgt = load_v2(v2_dir, v2_src_name, v2_tgt_name)
    sub_src, sub_tgt = nested_v2_subset(v2_src, v2_tgt, alpha, subset_seed)

    comp_src = list(v1_src) + list(sub_src)
    comp_tgt = list(v1_tgt) + list(sub_tgt)
    reps = max(1, int(repetitions))

    meta = Composition(
        alpha=float(alpha),
        repetitions=reps,
        n_v1=len(v1_src),
        n_v2_pool=len(v2_src),
        n_v2_used=len(sub_src),
        n_composed=len(comp_src),
        training_examples=len(comp_src) * reps,
        subset_seed=int(subset_seed),
    )
    return comp_src, comp_tgt, meta
