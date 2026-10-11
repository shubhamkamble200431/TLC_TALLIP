"""
Dataset + sampler for TLC_TALLIP (Sanskrit -> Hindi only).

Derived from TLC_multi/data/dataset.py.  Differences required by the brief:

  * TLC "control token" is switchable independently of the length loss
    (`use_length_control_token`) — section 19;
  * every item carries `bucket` (source-length bucket) and `target_length`
    (reference token count) so the differentiable length loss can use them —
    section 20;
  * NLLB and mBART-50 tokenizer language handling only (mT5 removed);
  * the bucket-balanced sampler is deterministic given a seed — section 4.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from src.data.buckets import DEFAULT_BOUNDARIES, assign_bucket, n_tokens
from src.utils.io import read_lines

logger = logging.getLogger(__name__)

LENGTH_CONTROL_TOKENS = [f"<LEN_BUCKET_{i}>" for i in range(1, 6)]


def load_parallel_corpus(data_dir, src_file, tgt_file, max_samples: Optional[int] = None):
    import os

    src = read_lines(os.path.join(data_dir, src_file))
    tgt = read_lines(os.path.join(data_dir, tgt_file))
    if len(src) != len(tgt):
        raise ValueError(
            f"Line-count mismatch: {src_file}={len(src)} vs {tgt_file}={len(tgt)}"
        )
    if max_samples:
        src, tgt = src[:max_samples], tgt[:max_samples]
    logger.info("Loaded %s pairs from %s (%s / %s)", f"{len(src):,}", data_dir, src_file, tgt_file)
    return src, tgt


class SaHiDataset(Dataset):
    def __init__(
        self,
        src_sentences: List[str],
        tgt_sentences: List[str],
        tokenizer,
        model_family: str = "nllb",
        tok_src_lang: Optional[str] = None,
        tok_tgt_lang: Optional[str] = None,
        max_input_length: int = 512,
        max_target_length: int = 128,
        use_length_control_token: bool = False,
        bucket_boundaries: List[int] = None,
        bucket_mode: str = "source_length",
        target_length_ratio: float = 1.8017,
        predicted_target_boundaries: List[float] = None,
        oracle_ratio_boundaries: List[float] = None,
    ):
        assert len(src_sentences) == len(tgt_sentences)
        self.src = src_sentences
        self.tgt = tgt_sentences
        self.tok = tokenizer
        self.family = model_family
        self.tok_src_lang = tok_src_lang
        self.tok_tgt_lang = tok_tgt_lang
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.use_length_control_token = use_length_control_token
        self.bucket_mode = bucket_mode
        self.boundaries = bucket_boundaries or DEFAULT_BOUNDARIES

        self._indic_processor = None
        if model_family == "indictrans2":
            from IndicTransToolkit import IndicProcessor

            self._indic_processor = IndicProcessor(inference=True)

        if bucket_mode == "source_length":
            self.buckets = [assign_bucket(n_tokens(s), self.boundaries) for s in src_sentences]
        elif bucket_mode == "predicted_target_length":
            pt_bounds = predicted_target_boundaries or [9, 18, 27, 36]
            self.buckets = [assign_bucket(round(n_tokens(s) * target_length_ratio), pt_bounds)
                            for s in src_sentences]
        elif bucket_mode == "oracle_ratio_class":
            ratio_bounds = oracle_ratio_boundaries or [1.0, 1.4, 1.667, 2.0]
            self.buckets = [assign_bucket(n_tokens(t) / max(n_tokens(s), 1), ratio_bounds)
                            for s, t in zip(src_sentences, tgt_sentences)]
        else:
            raise ValueError(f"unknown bucket_mode: {bucket_mode!r}")

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx: int) -> Dict:
        src = self.src[idx]
        tgt = self.tgt[idx]
        bucket = self.buckets[idx]

        if self._indic_processor is not None:
            # IndicTrans2's custom tokenizer expects pre-tagged source text
            # ("<src_lang> <tgt_lang> sentence") and normalized (untagged)
            # target text -- see src/models/builder.py's indictrans2 branch
            # and IndicTransToolkit.IndicProcessor.preprocess_batch.
            src = self._indic_processor.preprocess_batch(
                [src], src_lang=self.tok_src_lang, tgt_lang=self.tok_tgt_lang, is_target=False)[0]
            tgt = self._indic_processor.preprocess_batch(
                [tgt], src_lang=self.tok_tgt_lang, tgt_lang=self.tok_tgt_lang, is_target=True)[0]

        # NLLB / mBART: set the tokenizer source language before encoding.
        if self.tok_src_lang is not None and hasattr(self.tok, "src_lang"):
            self.tok.src_lang = self.tok_src_lang
        if self.tok_tgt_lang is not None and hasattr(self.tok, "tgt_lang"):
            self.tok.tgt_lang = self.tok_tgt_lang

        model_inputs = self.tok(
            src, max_length=self.max_input_length, truncation=True, padding=False
        )

        tgt_text = f"<LEN_BUCKET_{bucket}> {tgt}" if self.use_length_control_token else tgt
        labels = self.tok(
            text_target=tgt_text,
            max_length=self.max_target_length,
            truncation=True,
            padding=False,
        )
        pad_id = self.tok.pad_token_id
        label_ids = [(t if t != pad_id else -100) for t in labels["input_ids"]]
        target_length = sum(1 for t in label_ids if t != -100)

        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "labels": label_ids,
            "bucket": bucket,
            "target_length": target_length,
        }


def stratified_subsample(src: List[str], tgt: List[str], n: int, boundaries: List[int],
                         seed: int) -> tuple:
    """
    Take ~`n` pairs while preserving the source-length bucket proportions of the
    FULL corpus (deterministic given `seed`).  Use this instead of a plain
    prefix `src[:n]` whenever the corpus might not already be shuffled, or when
    a subsample must match the parent distribution exactly (e.g. the BBS search,
    which is entirely about bucket proportions).
    """
    import random as _random

    from src.data.buckets import assign_bucket, n_tokens

    if n >= len(src):
        return list(src), list(tgt)
    by_b = {}
    for i, s in enumerate(src):
        by_b.setdefault(assign_bucket(n_tokens(s), boundaries), []).append(i)
    rng = _random.Random(int(seed))
    keep = []
    total = len(src)
    for b in sorted(by_b):
        idx = by_b[b][:]
        rng.shuffle(idx)
        take = round(n * len(idx) / total)
        keep.extend(idx[:take])
    # fix rounding drift back to exactly n
    if len(keep) > n:
        rng.shuffle(keep)
        keep = keep[:n]
    elif len(keep) < n:
        remaining = [i for i in range(total) if i not in set(keep)]
        rng.shuffle(remaining)
        keep.extend(remaining[: n - len(keep)])
    keep.sort()
    return [src[i] for i in keep], [tgt[i] for i in keep]


def normalized_bucket_weights(bucket_weights: List[float]) -> List[float]:
    """Mean-normalise so sum(w) = len(w)  <=>  mean(w) = 1 (brief section 6)."""
    w = [float(x) for x in bucket_weights]
    m = sum(w) / len(w)
    if m <= 0:
        raise ValueError(f"bucket_weights must have positive mean, got {bucket_weights}")
    return [x / m for x in w]


def build_weighted_sampler(
    dataset: SaHiDataset,
    bucket_weights: List[float],
    seed: int,
    deterministic: bool = True,
) -> WeightedRandomSampler:
    """
    WeightedRandomSampler over per-example weights derived from the (mean-
    normalised) bucket weights.  A dedicated seeded torch.Generator makes BBS
    sampling reproducible for a given seed (brief section 4).
    """
    norm = normalized_bucket_weights(bucket_weights)
    weights = torch.tensor([norm[b - 1] for b in dataset.buckets], dtype=torch.double)
    generator = None
    if deterministic:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )
