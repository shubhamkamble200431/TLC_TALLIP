"""
Paired bootstrap significance testing with Holm-Bonferroni (brief section 17).

  * resampling unit = the sentence-level (hypothesis, reference[, source]) tuple
    -- never individual tokens;
  * 10,000 bootstrap resamples by default;
  * 95% CI on the metric difference (system A - system B);
  * two-sided bootstrap p-value;
  * Holm-Bonferroni correction for a family of pairwise comparisons.

Efficiency: for corpus-level metrics (BLEU / chrF2 / TER) the per-sentence
sacrebleu sufficient statistics are extracted ONCE; each resample is then a
vector sum + one score computation, so 10k resamples over the 7.2k-sentence
test set run in seconds rather than re-tokenising every time.  `metric =
"precomputed"` bootstraps the mean of caller-supplied segment scores (e.g.
COMET / BERTScore segment outputs).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    from sacrebleu.metrics import BLEU, CHRF, TER

    _SB = True
except Exception:  # pragma: no cover
    _SB = False


_METRIC_CACHE: Dict[str, object] = {}


def _metric_obj(metric: str):
    """Cached sacrebleu metric object (constructing BLEU reloads a tokenizer)."""
    if metric not in _METRIC_CACHE:
        if metric == "bleu":
            _METRIC_CACHE[metric] = BLEU(tokenize="flores200", effective_order=False)
        elif metric == "chrf2":
            _METRIC_CACHE[metric] = CHRF(word_order=2)
        elif metric == "ter":
            _METRIC_CACHE[metric] = TER()
        else:
            raise ValueError(f"unknown corpus metric {metric!r}")
    return _METRIC_CACHE[metric]


def _per_sentence_stats(metric: str, hyps: Sequence[str], refs: Sequence[str]) -> np.ndarray:
    """[n_sentences, n_stats] sufficient statistics for `metric`."""
    m = _metric_obj(metric)
    rows = [m._extract_corpus_statistics([h], [[r]])[0] for h, r in zip(hyps, refs)]
    return np.asarray(rows, dtype=float)


def _score_from_stats(metric: str, summed: np.ndarray) -> float:
    return float(_metric_obj(metric)._compute_score_from_stats(list(summed)).score)


@dataclass
class BootstrapResult:
    metric: str
    n_items: int
    n_samples: int
    score_a: float
    score_b: float
    observed_diff: float
    bootstrap_mean_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    comparison: str = ""
    p_value_corrected: Optional[float] = None
    reject_null: Optional[bool] = None

    def to_dict(self) -> Dict:
        return self.__dict__.copy()


def paired_bootstrap(
    refs: Sequence[str],
    sys_a: Sequence[str],
    sys_b: Sequence[str],
    metric: str = "bleu",
    n_samples: int = 10_000,
    seed: int = 42,
    seg_scores_a: Optional[Sequence[float]] = None,
    seg_scores_b: Optional[Sequence[float]] = None,
    comparison: str = "",
) -> BootstrapResult:
    n = len(refs)
    if not (len(sys_a) == len(sys_b) == n):
        raise ValueError("refs, sys_a, sys_b must be the same length")
    rng = np.random.default_rng(seed)
    idx_matrix = rng.integers(0, n, size=(n_samples, n))

    if metric == "precomputed":
        if seg_scores_a is None or seg_scores_b is None:
            raise ValueError("metric='precomputed' needs seg_scores_a and seg_scores_b")
        a = np.asarray(seg_scores_a, dtype=float)
        b = np.asarray(seg_scores_b, dtype=float)
        score_a, score_b = float(a.mean()), float(b.mean())
        diffs = a[idx_matrix].mean(axis=1) - b[idx_matrix].mean(axis=1)
    else:
        if not _SB:
            raise RuntimeError("sacrebleu required for corpus-level bootstrap metrics")
        stats_a = _per_sentence_stats(metric, sys_a, refs)
        stats_b = _per_sentence_stats(metric, sys_b, refs)
        score_a = _score_from_stats(metric, stats_a.sum(axis=0))
        score_b = _score_from_stats(metric, stats_b.sum(axis=0))
        # one vectorised gather+sum for all resamples, then score each summed row
        sum_a = stats_a[idx_matrix].sum(axis=1)   # [n_samples, n_stats]
        sum_b = stats_b[idx_matrix].sum(axis=1)
        diffs = np.array([
            _score_from_stats(metric, sum_a[i]) - _score_from_stats(metric, sum_b[i])
            for i in range(n_samples)
        ], dtype=float)

    observed = score_a - score_b
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    if observed >= 0:
        p = 2.0 * float(np.mean(diffs <= 0.0))
    else:
        p = 2.0 * float(np.mean(diffs >= 0.0))
    p = min(1.0, p)

    return BootstrapResult(
        metric=metric,
        n_items=n,
        n_samples=n_samples,
        score_a=score_a,
        score_b=score_b,
        observed_diff=observed,
        bootstrap_mean_diff=float(diffs.mean()),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=p,
        comparison=comparison or f"A_vs_B[{metric}]",
    )


def holm_bonferroni(results: List[BootstrapResult], alpha: float = 0.05) -> List[BootstrapResult]:
    """In-place Holm-Bonferroni across a family of comparisons (brief section 17)."""
    order = sorted(range(len(results)), key=lambda i: results[i].p_value)
    m = len(results)
    prev = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * results[i].p_value)
        adj = max(adj, prev)
        prev = adj
        results[i].p_value_corrected = adj
        results[i].reject_null = adj < alpha
    return results
