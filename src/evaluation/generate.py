"""
Unified generation interface (brief section 22).

`translate()` works identically for NLLB and mBART-50: it sets the tokenizer
source language, forces the decoder BOS to the target language where the
architecture requires it, and returns a plain list of strings.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


def translate(
    model,
    tokenizer,
    sources: List[str],
    *,
    family: str = "nllb",
    tok_src_lang: Optional[str] = None,
    tok_tgt_lang: Optional[str] = None,
    forced_bos_token_id: Optional[int] = None,
    batch_size: int = 32,
    max_input_length: int = 512,
    max_new_tokens: int = 128,
    num_beams: int = 4,
    no_repeat_ngram_size: int = 0,
    device: str = "cuda",
) -> List[str]:
    model.eval()
    if tok_src_lang is not None and hasattr(tokenizer, "src_lang"):
        tokenizer.src_lang = tok_src_lang

    indic_processor = None
    if family == "indictrans2":
        from IndicTransToolkit import IndicProcessor

        indic_processor = IndicProcessor(inference=True)

    preds: List[str] = []
    n_batches = (len(sources) + batch_size - 1) // batch_size
    for bi, start in enumerate(range(0, len(sources), batch_size), 1):
        batch = sources[start : start + batch_size]
        if indic_processor is not None:
            # tag + normalize source text the same way training does (see
            # src/data/dataset.py::SaHiDataset) -- IndicTrans2's tokenizer
            # requires "<src_lang> <tgt_lang> sentence" formatted input.
            batch = indic_processor.preprocess_batch(
                batch, src_lang=tok_src_lang, tgt_lang=tok_tgt_lang, is_target=False)
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        ).to(device)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
        )
        if no_repeat_ngram_size:
            gen_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        if forced_bos_token_id is not None:
            gen_kwargs["forced_bos_token_id"] = forced_bos_token_id
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
        if indic_processor is not None:
            # restore placeholders, fix script issues, detokenize -- the
            # counterpart of the preprocess_batch call above.
            decoded = indic_processor.postprocess_batch(decoded, lang=tok_tgt_lang or "hin_Deva")
        preds.extend(decoded)
        if bi % 20 == 0 or bi == n_batches:
            logger.info("  generated %d/%d", min(start + batch_size, len(sources)), len(sources))
    return [p.strip() for p in preds]
