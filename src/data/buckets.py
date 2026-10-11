"""
Source-length bucket assignment (brief sections 6, 15).

Natural buckets on the *source* side, whitespace-token counted:

    B1 = 1-5      B2 = 6-10     B3 = 11-15    B4 = 16-20    B5 = 21+

Brief section 15: "Use the natural length buckets rather than artificially
moving short sentences into long buckets."  Nothing here rebalances.
"""
from __future__ import annotations

from typing import List

DEFAULT_BOUNDARIES: List[int] = [5, 10, 15, 20]

# Reviewer ablation (Q3): does TLC help because it exposes genuine TARGET-length
# signal, or just because it makes already-available SOURCE length explicit?
# Two alternative bucketing schemes, same 5-bucket structure, different value
# being bucketed -- see src/data/dataset.py::SaHiDataset bucket_mode.
#   predicted_target_length: round(source_len * GLOBAL_TARGET_LENGTH_RATIO),
#     boundaries scaled by the same ratio -- deployable at inference (needs
#     only the source sentence, same as the current scheme).
#   oracle_ratio_class: true target_len / source_len, boundaries = train-set
#     quintiles -- TRAIN-TIME ONLY (needs the true target length), an oracle
#     upper bound, not deployable at inference without a separate predictor.
GLOBAL_TARGET_LENGTH_RATIO: float = 1.8017  # mean(target_len/source_len) over data/v2/train
PREDICTED_TARGET_BOUNDARIES: List[float] = [9, 18, 27, 36]       # DEFAULT_BOUNDARIES * ratio, rounded
ORACLE_RATIO_BOUNDARIES: List[float] = [1.0, 1.4, 1.667, 2.0]    # train-set ratio quintile cutoffs

BUCKET_IDS = [1, 2, 3, 4, 5]
BUCKET_LABELS = {1: "B1", 2: "B2", 3: "B3", 4: "B4", 5: "B5"}
BUCKET_RANGES = {1: "1-5", 2: "6-10", 3: "11-15", 4: "16-20", 5: "21+"}
BUCKET_LABELS_LONG = {b: f"{BUCKET_LABELS[b]}_{BUCKET_RANGES[b]}" for b in BUCKET_IDS}


def n_tokens(text: str) -> int:
    return len(text.split())


def assign_bucket(num_tokens: int, boundaries: List[int] = DEFAULT_BOUNDARIES) -> int:
    """Map a source token count to a bucket id in 1..len(boundaries)+1."""
    for i, b in enumerate(boundaries):
        if num_tokens <= b:
            return i + 1
    return len(boundaries) + 1


def bucket_of_sentence(text: str, boundaries: List[int] = DEFAULT_BOUNDARIES) -> int:
    return assign_bucket(n_tokens(text), boundaries)


def bucket_series(sentences, boundaries: List[int] = DEFAULT_BOUNDARIES) -> List[int]:
    return [bucket_of_sentence(s, boundaries) for s in sentences]


def bucket_counts(sentences, boundaries: List[int] = DEFAULT_BOUNDARIES) -> dict:
    counts = {b: 0 for b in BUCKET_IDS}
    for s in sentences:
        counts[assign_bucket(n_tokens(s), boundaries)] += 1
    return counts
