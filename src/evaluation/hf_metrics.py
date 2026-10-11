"""
`compute_metrics` for the HF Seq2SeqTrainer's *validation* passes.

Only validation-set metrics feed `metric_for_best_model` / checkpoint
selection (brief section 28: the test set is never touched here).  COMET is
run every N validation calls to keep training cheap.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from src.evaluation.metrics import corpus_bleu, corpus_chrf2, corpus_comet
from src.evaluation.metrics import length_metrics

logger = logging.getLogger(__name__)


def make_compute_metrics(
    tokenizer,
    sources: Optional[List[str]] = None,
    comet_every: int = 5,
    run_comet: bool = True,
):
    state = {"calls": 0}

    def compute_metrics(eval_preds):
        state["calls"] += 1
        try:
            pred_ids, label_ids = eval_preds
            if isinstance(pred_ids, tuple):
                pred_ids = pred_ids[0]
            pred_ids = np.where(pred_ids != -100, pred_ids, tokenizer.pad_token_id)
            label_ids = np.where(label_ids != -100, label_ids, tokenizer.pad_token_id)
            preds = [p.strip() for p in tokenizer.batch_decode(pred_ids, skip_special_tokens=True)]
            refs = [l.strip() for l in tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

            out = {"bleu": corpus_bleu(preds, refs), "chrf2": corpus_chrf2(preds, refs)}
            lm = length_metrics(preds, refs)
            out["length_ratio"] = lm.get("length_ratio", float("nan"))
            out["length_error"] = lm.get("absolute_length_error", float("nan"))

            do_comet = run_comet and sources is not None and (
                state["calls"] == 1 or state["calls"] % max(comet_every, 1) == 0
            )
            if do_comet:
                out["comet"] = corpus_comet(preds, refs, sources[: len(preds)])
            return out
        except Exception as exc:  # keep training alive
            logger.error("[compute_metrics] failed: %s", exc, exc_info=True)
            return {"bleu": 0.0, "chrf2": 0.0}

    return compute_metrics
