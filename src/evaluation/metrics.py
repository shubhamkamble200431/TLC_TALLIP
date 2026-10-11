"""
Unified metric computation (brief sections 14, 15, 22).

Every model (NLLB, mBART-50) produces plain ``predictions`` / ``references`` /
``sources`` lists and this module scores them — no model-specific metric code.

Quality metrics : BLEU, chrF2, TER, METEOR, BERTScore-F1, COMET
Length metrics  : mean abs length error, relative length error, length ratio,
                  premature-EOS rate, over-generation rate, truncation rate
Bucketed        : all of the above per source-length bucket B1..B5
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from src.data.buckets import DEFAULT_BOUNDARIES, BUCKET_IDS, BUCKET_LABELS_LONG, assign_bucket, n_tokens

logger = logging.getLogger(__name__)

try:
    import sacrebleu

    _SACREBLEU = True
except Exception:  # pragma: no cover
    _SACREBLEU = False

_COMET_MODEL = None
_METEOR_READY = False


# ---------------------------------------------------------------------------
# quality metrics
# ---------------------------------------------------------------------------
def corpus_bleu(preds: List[str], refs: List[str]) -> float:
    if not _SACREBLEU or not preds:
        return float("nan")
    return sacrebleu.corpus_bleu(preds, [refs], tokenize="flores200").score


def corpus_chrf2(preds: List[str], refs: List[str]) -> float:
    if not _SACREBLEU or not preds:
        return float("nan")
    return sacrebleu.corpus_chrf(preds, [refs], word_order=2).score


def corpus_ter(preds: List[str], refs: List[str]) -> float:
    if not _SACREBLEU or not preds:
        return float("nan")
    return sacrebleu.corpus_ter(preds, [refs]).score


def corpus_meteor(preds: List[str], refs: List[str]) -> float:
    global _METEOR_READY
    if not preds:
        return float("nan")
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score

        if not _METEOR_READY:
            for pkg in ("wordnet", "omw-1.4", "punkt"):
                try:
                    nltk.download(pkg, quiet=True)
                except Exception:
                    pass
            _METEOR_READY = True
        scores = [meteor_score([r.split()], p.split()) for p, r in zip(preds, refs)]
        return float(np.mean(scores)) * 100.0
    except Exception as exc:  # pragma: no cover
        logger.warning("METEOR unavailable: %s", exc)
        return float("nan")


def corpus_bertscore(preds: List[str], refs: List[str], lang: str = "hi") -> float:
    if not preds:
        return float("nan")
    try:
        from bert_score import score as bertscore

        _, _, f1 = bertscore(preds, refs, lang=lang, verbose=False)
        return float(f1.mean().item()) * 100.0
    except Exception as exc:  # pragma: no cover
        logger.warning("BERTScore unavailable: %s", exc)
        return float("nan")


def _get_comet():
    global _COMET_MODEL
    if _COMET_MODEL is None:
        from comet import download_model, load_from_checkpoint

        _COMET_MODEL = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
    return _COMET_MODEL


def corpus_comet(preds: List[str], refs: List[str], srcs: List[str], gpus: int = 1) -> float:
    if not preds:
        return float("nan")
    try:
        model = _get_comet()
        data = [{"src": s, "mt": p, "ref": r} for s, p, r in zip(srcs, preds, refs)]
        import torch

        out = model.predict(data, batch_size=64, gpus=gpus if torch.cuda.is_available() else 0)
        return float(np.mean(out.scores)) * 100.0
    except Exception as exc:  # pragma: no cover
        logger.warning("COMET failed: %s", exc)
        return float("nan")


# ---------------------------------------------------------------------------
# length metrics  (brief section 14 / 15)
# ---------------------------------------------------------------------------
def length_metrics(
    preds: List[str],
    refs: List[str],
    tol: float = 0.2,
    max_gen_len: Optional[int] = None,
) -> Dict[str, float]:
    if not preds:
        return {}
    pl = np.array([n_tokens(p) for p in preds], dtype=float)
    rl = np.array([max(n_tokens(r), 1) for r in refs], dtype=float)
    abs_err = np.abs(pl - rl)
    rel_err = abs_err / rl
    ratio = pl / rl
    premature = float(np.mean(pl < rl * (1.0 - tol)))     # stopped too early
    over = float(np.mean(pl > rl * (1.0 + tol)))          # ran on too long
    if max_gen_len is not None:
        trunc = float(np.mean(pl >= max_gen_len))
    else:
        trunc = float("nan")
    return {
        "absolute_length_error": float(abs_err.mean()),
        "relative_length_error": float(rel_err.mean()),
        "length_ratio": float(ratio.mean()),
        "premature_eos_rate": premature,
        "overgeneration_rate": over,
        "truncation_rate": trunc,
    }


def degenerate_flags(preds: List[str]) -> Dict[str, float]:
    """Cheap failure signals (brief section 26)."""
    n = max(len(preds), 1)
    empty = sum(1 for p in preds if not p.strip())
    return {
        "zero_length_prediction_rate": empty / n,
        "n_zero_length_predictions": empty,
    }


# ---------------------------------------------------------------------------
def compute_all(
    preds: List[str],
    refs: List[str],
    srcs: Optional[List[str]] = None,
    boundaries: List[int] = DEFAULT_BOUNDARIES,
    metrics: Optional[List[str]] = None,
    run_comet: bool = True,
    max_gen_len: Optional[int] = None,
    bertscore_lang: str = "hi",
    comet_gpus: int = 1,
) -> Dict[str, float]:
    """Overall + per-bucket metrics as a single flat dict."""
    if len(preds) != len(refs):
        raise ValueError(f"preds ({len(preds)}) != refs ({len(refs)})")
    wanted = set(metrics or ["bleu", "chrf2", "ter", "meteor", "bertscore", "comet"])
    res: Dict[str, float] = {"n_examples": len(preds)}

    def _score_block(p, r, s, prefix=""):
        b = {}
        if "bleu" in wanted:
            b[prefix + "bleu"] = corpus_bleu(p, r)
        if "chrf2" in wanted:
            b[prefix + "chrf2"] = corpus_chrf2(p, r)
        if "ter" in wanted:
            b[prefix + "ter"] = corpus_ter(p, r)
        if "meteor" in wanted:
            b[prefix + "meteor"] = corpus_meteor(p, r)
        if "bertscore" in wanted:
            b[prefix + "bertscore"] = corpus_bertscore(p, r, lang=bertscore_lang)
        if "comet" in wanted and run_comet and s is not None:
            b[prefix + "comet"] = corpus_comet(p, r, s, gpus=comet_gpus)
        b.update({prefix + k: v for k, v in length_metrics(p, r, max_gen_len=max_gen_len).items()})
        return b

    res.update(_score_block(preds, refs, srcs))
    res.update(degenerate_flags(preds))

    # per-bucket (natural membership on the SOURCE side)
    if srcs is not None:
        buckets = [assign_bucket(n_tokens(s), boundaries) for s in srcs]
    else:
        buckets = [assign_bucket(n_tokens(r), boundaries) for r in refs]
    for b in BUCKET_IDS:
        idx = [i for i, bk in enumerate(buckets) if bk == b]
        label = BUCKET_LABELS_LONG[b]
        res[f"n_examples_{label}"] = len(idx)
        if not idx:
            continue
        bp = [preds[i] for i in idx]
        br = [refs[i] for i in idx]
        bs = [srcs[i] for i in idx] if srcs is not None else None
        res.update(_score_block(bp, br, bs, prefix=f"{label}_"))
    return res
