"""
Model + tokenizer construction for NLLB and mBART-50 (mT5 removed).

Returns a `ModelBundle` carrying everything the trainer / evaluator need so
that no downstream code has to branch on model family (brief section 22:
"Create a unified interface ... Do not write model-specific metric code").
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from src.config import RunConfig
from src.models.lang_codes import infer_model_family, nllb_to_mbart50
from src.models.positional import apply_alibi_to_model, apply_rope_to_model

logger = logging.getLogger(__name__)

LENGTH_CONTROL_TOKENS = [f"<LEN_BUCKET_{i}>" for i in range(1, 6)]


@dataclass
class ModelBundle:
    model: nn.Module
    tokenizer: object
    family: str                      # "nllb" | "mbart"
    tok_src_lang: Optional[str]
    tok_tgt_lang: Optional[str]
    forced_bos_token_id: Optional[int]
    eos_token_id: int
    rope_layers: int = 0
    alibi_layers: int = 0
    added_length_tokens: int = 0


# ---------------------------------------------------------------------------
def _rematerialize_tied_embeddings(model: nn.Module, base_model: str) -> None:
    """Workaround for a transformers/accelerate bug that leaves the tied
    input/output embedding on the meta device after from_pretrained
    (observed on tied-embedding checkpoints, transformers 4.57.x)."""
    meta = [n for n, p in model.named_parameters() if p.is_meta]
    if not meta:
        return
    from safetensors import safe_open
    from transformers.utils import cached_file

    ckpt = cached_file(base_model, "model.safetensors")
    with safe_open(ckpt, framework="pt") as f:
        real = f.get_tensor("lm_head.weight").to(model.dtype)
    old = model.get_output_embeddings().weight
    new = nn.Parameter(real)
    for m in model.modules():
        if getattr(m, "weight", None) is old:
            m.weight = new
    still = [n for n, p in model.named_parameters() if p.is_meta]
    if still:
        raise RuntimeError(f"meta params remain after rematerialization: {still}")
    logger.warning("[builder] rematerialized tied embedding left on meta device (%s)", base_model)


def _load_tokenizer(cfg: RunConfig, family: str):
    if family == "mbart":
        from transformers import MBart50TokenizerFast

        src = cfg.mbart_src_lang or nllb_to_mbart50(cfg.src_lang)
        tgt = cfg.mbart_tgt_lang or nllb_to_mbart50(cfg.tgt_lang)
        tok = MBart50TokenizerFast.from_pretrained(cfg.base_model, src_lang=src, tgt_lang=tgt)
        return tok, src, tgt
    if family == "mt5":
        # T5-family tokenizer has no src_lang/tgt_lang concept — plain fine-tune,
        # source text in, target text out (no task prefix; see baseline_mt5.yaml).
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
        return tok, None, None
    if family == "indictrans2":
        # IndicTrans2's own tokenizer (custom_code repo; gated — needs HF auth +
        # accepted license).  Language tags are plain-text prefixes it inserts
        # itself, not tokenizer.src_lang/tgt_lang, so none are returned here.
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True, use_fast=True)
        return tok, cfg.src_lang, cfg.tgt_lang
    if family in ("tf6", "tf12", "tf18"):
        # from-scratch Transformer: borrow a pretrained tokenizer's vocab (NLLB
        # by default via tokenizer_base) purely for subword segmentation.
        from transformers import AutoTokenizer

        tok_src_model = cfg.tokenizer_base or cfg.base_model
        tok = AutoTokenizer.from_pretrained(
            tok_src_model, use_fast=True, src_lang=cfg.src_lang, tgt_lang=cfg.tgt_lang
        )
        return tok, cfg.src_lang, cfg.tgt_lang
    # NLLB
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        cfg.base_model, use_fast=True, src_lang=cfg.src_lang, tgt_lang=cfg.tgt_lang
    )
    return tok, cfg.src_lang, cfg.tgt_lang


def _build_scratch_transformer(cfg: RunConfig, tokenizer, n_layers: int) -> nn.Module:
    """A vanilla from-scratch encoder-decoder Transformer (BART architecture,
    randomly initialised) sized by scratch_* in the model YAML.  Reuses
    `tokenizer`'s vocab (see tokenizer_base) purely for subword segmentation —
    there is no pretrained checkpoint for tf6/tf12."""
    from transformers import BartConfig, BartForConditionalGeneration

    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    mconf = BartConfig(
        vocab_size=len(tokenizer),
        d_model=cfg.scratch_d_model,
        encoder_layers=n_layers, decoder_layers=n_layers,
        encoder_attention_heads=cfg.scratch_attention_heads,
        decoder_attention_heads=cfg.scratch_attention_heads,
        encoder_ffn_dim=cfg.scratch_ffn_dim, decoder_ffn_dim=cfg.scratch_ffn_dim,
        max_position_embeddings=max(cfg.max_input_length, cfg.max_target_length) + 8,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=bos,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.eos_token_id,
        forced_eos_token_id=tokenizer.eos_token_id,
        dropout=0.1, attention_dropout=0.1,
    )
    return BartForConditionalGeneration(mconf)


def _load_model(cfg: RunConfig, family: str, tokenizer=None) -> nn.Module:
    kwargs = {"dtype": torch.float32}
    if cfg.model_revision:
        kwargs["revision"] = cfg.model_revision
    if cfg.use_alibi:
        # ALiBi injects a per-head additive bias into the attention mask; the
        # SDPA/flash backends require a last-dim-contiguous mask and HF slices
        # the mask internally, breaking that.  The eager attention path adds the
        # mask with plain broadcasting and has no such constraint.
        kwargs["attn_implementation"] = "eager"
    if family == "mbart":
        from transformers import MBartForConditionalGeneration

        model = MBartForConditionalGeneration.from_pretrained(cfg.base_model, **kwargs)
    elif family == "indictrans2":
        from transformers import AutoModelForSeq2SeqLM

        model = AutoModelForSeq2SeqLM.from_pretrained(cfg.base_model, trust_remote_code=True, **kwargs)
        # IndicTrans2's custom (trust_remote_code) decoder indexes
        # past_key_values[0][0].shape[2] assuming the legacy tuple-of-tuples
        # cache format; this transformers version's generate() passes a
        # Cache object instead, so past_key_values[0] is None and the custom
        # code crashes (AttributeError: 'NoneType' object has no attribute
        # 'shape'). Disabling the KV cache sidesteps the incompatible code
        # path entirely (slower generation, full recompute each step) --
        # same fix already applied to ALiBi's custom attention path below.
        model.config.use_cache = False
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.use_cache = False
    elif family in ("tf6", "tf12", "tf18"):
        model = _build_scratch_transformer(cfg, tokenizer, cfg.scratch_layers)
    else:
        from transformers import AutoModelForSeq2SeqLM

        model = AutoModelForSeq2SeqLM.from_pretrained(cfg.base_model, **kwargs)
    _rematerialize_tied_embeddings(model, cfg.base_model)
    return model


def _resolve_forced_bos(tokenizer, family: str, tok_tgt_lang: str, cfg: RunConfig) -> Optional[int]:
    if not cfg.force_bos_target_lang:
        return None
    # newer transformers: prefer convert_tokens_to_ids; fall back to lang_code_to_id
    tid = None
    try:
        tid = tokenizer.convert_tokens_to_ids(tok_tgt_lang)
    except Exception:
        pass
    if (tid is None or tid == tokenizer.unk_token_id) and hasattr(tokenizer, "lang_code_to_id"):
        tid = tokenizer.lang_code_to_id.get(tok_tgt_lang)
    return tid


# ---------------------------------------------------------------------------
def build(cfg: RunConfig) -> ModelBundle:
    family = infer_model_family(cfg.base_model)
    logger.info("[builder] family=%s base_model=%s", family, cfg.base_model)

    tokenizer, tok_src, tok_tgt = _load_tokenizer(cfg, family)
    model = _load_model(cfg, family, tokenizer=tokenizer)

    added = 0
    if cfg.use_length_control_token:
        new = [t for t in LENGTH_CONTROL_TOKENS if t not in tokenizer.get_vocab()]
        if new:
            tokenizer.add_special_tokens({"additional_special_tokens": new})
            model.resize_token_embeddings(len(tokenizer))
            added = len(new)
            logger.info("[builder] added length-control tokens: %s", new)
        if getattr(cfg, "bucket_embed_dim", None):
            from src.models.bucket_embedding import wrap_bucket_embedding

            bucket_ids = [tokenizer.convert_tokens_to_ids(t) for t in LENGTH_CONTROL_TOKENS]
            wrap_bucket_embedding(model, bucket_ids, int(cfg.bucket_embed_dim))
            logger.info("[builder] bucket tokens %s now use a dedicated dim=%d embedding+projection "
                       "(independent of d_model=%d)", bucket_ids, cfg.bucket_embed_dim, model.config.d_model)

    rope_layers = alibi_layers = 0
    mcfg = model.config
    n_heads = getattr(mcfg, "encoder_attention_heads", getattr(mcfg, "num_attention_heads", 16))
    head_dim = getattr(mcfg, "d_model", 1024) // max(n_heads, 1)
    if cfg.use_rope:
        rope_layers = apply_rope_to_model(model, head_dim=head_dim, max_seq_len=cfg.max_input_length)
    if cfg.use_alibi:
        alibi_layers = apply_alibi_to_model(model, num_heads=n_heads, max_seq_len=cfg.max_input_length)
        # The ALiBi bias is length-indexed on the (full) self-attention window;
        # incremental KV-cache decoding would feed it q_len=k_len=1.  Disable the
        # cache so generation stays correct (persisted into the checkpoint).
        model.config.use_cache = False
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.use_cache = False

    forced_bos = _resolve_forced_bos(tokenizer, family, tok_tgt, cfg)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = getattr(model.config, "eos_token_id", None)
    logger.info(
        "[builder] forced_bos_token_id=%s (tgt=%s)  eos_token_id=%s", forced_bos, tok_tgt, eos_id
    )

    # generation defaults on the model config (brief section 5: set explicitly)
    if forced_bos is not None:
        model.config.forced_bos_token_id = forced_bos
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.forced_bos_token_id = forced_bos

    total = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("[builder] %.1fM params  rope_layers=%d  alibi_layers=%d", total, rope_layers, alibi_layers)

    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        family=family,
        tok_src_lang=tok_src,
        tok_tgt_lang=tok_tgt,
        forced_bos_token_id=forced_bos,
        eos_token_id=int(eos_id),
        rope_layers=rope_layers,
        alibi_layers=alibi_layers,
        added_length_tokens=added,
    )
