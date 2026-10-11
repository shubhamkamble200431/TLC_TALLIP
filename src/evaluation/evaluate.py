"""
Held-out evaluation of a trained checkpoint (brief sections 15, 22, 28).

Every model is evaluated on BOTH test variants of BOTH corpora:

    <run_dir>/evaluation/<corpus>/<variant>/        corpus in {v2, v1}
        source references predictions                variant in {sampled, unsampled}
        metrics.json  bucket_rows.json  examples.txt

    sampled   = the bucket-balanced test set  (v2: test_7264_5 , v1: test_5414_5)
    unsampled = the raw test set              (test.sa / test.hi)

`evaluate_test_variants()` loads the checkpoint ONCE and loops over the grid.
`evaluate_checkpoint()` is the single-split helper used by the validation-only
selection paths (BBS / TLC search).
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence

import torch

from src.config import RunConfig, dataset_data_dir, dataset_test_variants
from src.data.buckets import BUCKET_IDS, BUCKET_LABELS_LONG, assign_bucket, n_tokens
from src.data.dataset import load_parallel_corpus
from src.evaluation.generate import translate
from src.evaluation.metrics import compute_all
from src.models.lang_codes import infer_model_family
from src.utils.io import ensure_dir, write_json, write_lines

logger = logging.getLogger(__name__)


def _load_ckpt(checkpoint_path: str, family: str, device: str, tokenizer_source: str = None):
    import json
    import os as _os

    from transformers import AutoTokenizer

    trc = family == "indictrans2"
    # IndicTrans2's custom tokenizer class breaks on save_pretrained ->
    # from_pretrained round-trips of itself (TypeError: __init__() got
    # multiple values for keyword argument 'src_vocab_file') -- load it from
    # its original hub repo instead. Safe whenever the vocab is unmodified
    # from pretraining (true for every indictrans2 baseline run: X-0000 only,
    # use_length_control_token=False, no tokens ever added).
    tok_path = tokenizer_source if (tokenizer_source and family == "indictrans2") else checkpoint_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=trc)
    if family == "mbart":
        from transformers import MBartForConditionalGeneration as M
    else:
        from transformers import AutoModelForSeq2SeqLM as M
    model = M.from_pretrained(checkpoint_path, torch_dtype=torch.float16, trust_remote_code=trc)

    bucket_marker = _os.path.join(checkpoint_path, "bucket_embed_dim.json")
    if _os.path.isfile(bucket_marker):
        # a plain from_pretrained (above) builds the STOCK architecture and
        # silently drops the trained BucketAwareEmbedding's weights as
        # "unused" -- re-apply the wrapper, then reload those specific
        # weights from the same checkpoint. See src/models/bucket_embedding.py
        # and the sidecar written in src/experiments/train_run.py.
        from src.models.bucket_embedding import wrap_bucket_embedding

        with open(bucket_marker) as f:
            marker = json.load(f)
        wrap_bucket_embedding(model, marker["bucket_token_ids"], marker["bucket_embed_dim"])
        safet = _os.path.join(checkpoint_path, "model.safetensors")
        if _os.path.isfile(safet):
            from safetensors.torch import load_file
            state = load_file(safet)
        else:
            state = torch.load(_os.path.join(checkpoint_path, "pytorch_model.bin"),
                               map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info("[builder] bucket_embed_dim=%d re-applied at eval load (missing=%d unexpected=%d)",
                   marker["bucket_embed_dim"], len(missing), len(unexpected))
        model = model.half()

    return model.to(device).eval(), tokenizer


def _forced_bos(cfg: RunConfig, tokenizer, family: str):
    if not cfg.force_bos_target_lang:
        return None
    tok_tgt = cfg.mbart_tgt_lang if family == "mbart" else cfg.tgt_lang
    fb = None
    try:
        fb = tokenizer.convert_tokens_to_ids(tok_tgt)
    except Exception:
        pass
    if (fb is None or fb == tokenizer.unk_token_id) and hasattr(tokenizer, "lang_code_to_id"):
        fb = tokenizer.lang_code_to_id.get(tok_tgt)
    return fb


def _score_and_write(
    cfg: RunConfig, model, tokenizer, family: str,
    srcs: List[str], refs: List[str], out_dir: str,
    *, num_beams: int, metrics, run_comet: bool, batch_size: int, device: str, meta: dict,
) -> Dict:
    ensure_dir(out_dir)
    tok_src = cfg.mbart_src_lang if family == "mbart" else cfg.src_lang
    tok_tgt = cfg.mbart_tgt_lang if family == "mbart" else cfg.tgt_lang
    preds = translate(
        model, tokenizer, srcs, family=family, tok_src_lang=tok_src, tok_tgt_lang=tok_tgt,
        forced_bos_token_id=_forced_bos(cfg, tokenizer, family),
        batch_size=batch_size, max_input_length=cfg.max_input_length,
        max_new_tokens=cfg.max_target_length, num_beams=num_beams, device=device,
    )
    write_lines(os.path.join(out_dir, "source"), srcs)
    write_lines(os.path.join(out_dir, "references"), refs)
    write_lines(os.path.join(out_dir, "predictions"), preds)

    scores = compute_all(
        preds, refs, srcs, boundaries=cfg.bucket_boundaries, metrics=metrics,
        run_comet=run_comet, max_gen_len=cfg.max_target_length, bertscore_lang="hi",
    )
    payload = {
        "case_id": cfg.case_id, "seed": cfg.seed,
        "dataset_version": cfg.dataset_version,
        "model_family": family, "model_label": getattr(cfg, "model_key", family),
        "checkpoint": meta.get("checkpoint"),
        "num_beams": num_beams,
        **meta,
        "metrics": scores,
    }
    write_json(os.path.join(out_dir, "metrics.json"), payload)

    bucket_rows = []
    for b in BUCKET_IDS:
        label = BUCKET_LABELS_LONG[b]
        row = {"bucket": f"B{b}", "n_examples": scores.get(f"n_examples_{label}", 0)}
        for m in ("bleu", "chrf2", "ter", "meteor", "bertscore", "comet",
                  "length_ratio", "absolute_length_error", "relative_length_error",
                  "premature_eos_rate", "overgeneration_rate", "truncation_rate"):
            if f"{label}_{m}" in scores:
                row[m] = scores[f"{label}_{m}"]
        bucket_rows.append(row)
    write_json(os.path.join(out_dir, "bucket_rows.json"), bucket_rows)
    _write_examples(out_dir, srcs, refs, preds, cfg.bucket_boundaries)
    return payload


def _write_examples(out_dir, srcs, refs, preds, boundaries, per_bucket=8):
    seen = {b: 0 for b in BUCKET_IDS}
    lines = []
    for s, r, p in zip(srcs, refs, preds):
        b = assign_bucket(n_tokens(s), boundaries)
        if seen[b] >= per_bucket:
            continue
        seen[b] += 1
        lines += [f"[B{b}] SRC: {s}", f"      REF: {r}", f"      HYP: {p}", ""]
    write_lines(os.path.join(out_dir, "examples.txt"), lines)


# ---------------------------------------------------------------------------
def evaluate_test_variants(
    cfg: RunConfig,
    checkpoint_path: str,
    corpora: Sequence[str] = ("v2",),
    variants: Sequence[str] = ("unsampled",),
    num_beams: Optional[int] = None,
    batch_size: int = 32,
    max_samples: Optional[int] = None,
    metrics: Optional[List[str]] = None,
    run_comet: bool = True,
    device: str = "cuda",
) -> Dict:
    """
    Evaluate `checkpoint_path` on {corpora} x {variants}, one folder each:
        <run_dir>/evaluation/<corpus>/<variant>/
    Returns {(corpus, variant): payload}.
    """
    family = infer_model_family(cfg.base_model)
    num_beams = num_beams or cfg.final_eval_num_beams
    model, tokenizer = _load_ckpt(checkpoint_path, family, device, tokenizer_source=cfg.base_model)
    out: Dict = {}
    try:
        for corpus in corpora:
            data_dir = dataset_data_dir(corpus)
            tv = dataset_test_variants(corpus)
            own = corpus == cfg.dataset_version
            for variant in variants:
                if variant not in tv or not tv[variant].get("src"):
                    logger.info("[eval] %s/%s not defined — skipping", corpus, variant)
                    continue
                src_file, tgt_file = tv[variant]["src"], tv[variant]["tgt"]
                srcs, refs = load_parallel_corpus(data_dir, src_file, tgt_file, max_samples=max_samples)
                out_dir = os.path.join(cfg.output_dir, "evaluation", corpus, variant)
                logger.info("[eval] %s/%s  (%d pairs)  -> %s", corpus, variant, len(srcs), out_dir)
                meta = {
                    "corpus": corpus, "variant": variant, "test_file": src_file,
                    "is_training_corpus": own,
                    "cross_corpus": not own,
                    "note": None if own else
                    "cross-corpus evaluation (training corpus != this test corpus); "
                    "see results/json/leakage_report.json for V1<->V2 overlap.",
                    "checkpoint": checkpoint_path,
                    "test_set_used_for_selection": False,
                }
                out[(corpus, variant)] = _score_and_write(
                    cfg, model, tokenizer, family, srcs, refs, out_dir,
                    num_beams=num_beams, metrics=metrics, run_comet=run_comet,
                    batch_size=batch_size, device=device, meta=meta,
                )
    finally:
        del model
        torch.cuda.empty_cache()
    return out


def evaluate_checkpoint(
    cfg: RunConfig,
    checkpoint_path: str,
    split: str = "val",
    out_dir: Optional[str] = None,
    num_beams: Optional[int] = None,
    batch_size: int = 32,
    max_samples: Optional[int] = None,
    metrics: Optional[List[str]] = None,
    run_comet: bool = True,
    device: str = "cuda",
) -> Dict:
    """Single-split eval against cfg.data_dir (used by the validation-only paths)."""
    family = infer_model_family(cfg.base_model)
    num_beams = num_beams or cfg.final_eval_num_beams
    out_dir = out_dir or os.path.join(cfg.output_dir, "evaluation", split)

    if split == "val":
        src_file, tgt_file = cfg.val_src, cfg.val_tgt
    elif split == "test":
        src_file, tgt_file = cfg.test_src, cfg.test_tgt
    elif split == "train":
        src_file, tgt_file = cfg.train_src, cfg.train_tgt
    else:
        raise ValueError(split)

    srcs, refs = load_parallel_corpus(cfg.data_dir, src_file, tgt_file, max_samples=max_samples)
    model, tokenizer = _load_ckpt(checkpoint_path, family, device, tokenizer_source=cfg.base_model)
    try:
        payload = _score_and_write(
            cfg, model, tokenizer, family, srcs, refs, out_dir,
            num_beams=num_beams, metrics=metrics, run_comet=run_comet,
            batch_size=batch_size, device=device,
            meta={"split": split, "test_file": src_file, "checkpoint": checkpoint_path},
        )
    finally:
        del model
        torch.cuda.empty_cache()
    return payload
