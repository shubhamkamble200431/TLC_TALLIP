"""
Core training orchestrator — one (case_id, seed, dataset) run.

Responsibilities
  * strict seeding (brief section 4) and a printed run banner (section 4);
  * dataset fingerprint verification (section 2 — no silent regeneration);
  * experiment manifest + frozen config.yaml (section 25);
  * resumable, interruption-safe training via StatusTracker (section 3 / 26);
  * a single held-out test evaluation AFTER the config is frozen (section 28);
  * automatic failure detection before the run is allowed to be 'completed'
    (section 26 / 27).
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional

from src.config import CONFIGS_DIR, RunConfig
from src.data.dataset import load_parallel_corpus
from src.data.manifest import verify_fingerprint
from src.evaluation.evaluate import evaluate_test_variants
from src.evaluation.hf_metrics import make_compute_metrics
from src.experiments import manifest as run_manifest
from src.experiments.failure import check_run
from src.experiments.status import StatusTracker, is_completed
from src.models.builder import build as build_model
from src.training.trainer import TALLIPTrainer, build_training_args
from src.utils.env_info import now_iso
from src.utils.io import ensure_dir, read_json
from src.utils.seeding import set_global_seed

logger = logging.getLogger(__name__)


def _load_dataset_manifest(dataset_version: str) -> dict:
    import yaml

    path = os.path.join(CONFIGS_DIR, "datasets", f"{dataset_version}.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _print_banner(cfg: RunConfig):
    f = cfg.factors
    logger.info("=" * 64)
    logger.info("  Seed:          %d", cfg.seed)
    logger.info("  Dataset:       %s", cfg.dataset_version)
    logger.info("  Configuration: %s", cfg.case_id)
    logger.info("  BBS:   %d", f["bbs"])
    logger.info("  TLC:   %d   (control_token=%s  length_loss=%s  lambda=%s)",
                f["tlc"], cfg.use_length_control_token, cfg.use_length_loss, cfg.length_loss_lambda)
    logger.info("  RoPE:  %d", f["rope"])
    logger.info("  ALiBi: %d", f["alibi"])
    logger.info("  Model: %s (%s)", cfg.base_model, cfg.model_family)
    logger.info("  Output dir:    %s", cfg.output_dir)
    logger.info("=" * 64)


class DoesNotFitError(RuntimeError):
    """Raised by _autoscale_micro_batch when even micro-batch 1 hits a hard
    CUDA OOM -- a real capacity problem, not something a retry fixes."""


def _latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    if not os.path.isdir(ckpt_dir):
        return None
    cks = [d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint-")]
    if not cks:
        return None
    cks.sort(key=lambda d: int(d.split("-")[-1]))
    return os.path.join(ckpt_dir, cks[-1])


def _autoscale_micro_batch(bundle, cfg, headroom: float = 0.20) -> None:
    """
    Pick the largest per-device micro-batch in {bs, bs/2, bs/4, ... , 1} that
    survives a real fwd+bwd+AdamW step within GPU memory (>= `headroom` free),
    then rescale gradient_accumulation_steps so bs*ga (the EFFECTIVE batch) is
    unchanged.  Mutates cfg in place.  No-op without CUDA.

    This replaces reliance on HF `auto_find_batch_size` catching a *hard* OOM:
    a config that merely fills the card (e.g. TLC at bs 32 -> ~38/40 GB) thrashes
    the CUDA allocator at ~zero throughput without ever raising OOM.  Probing up
    front with an optimiser step + a headroom margin avoids that and keeps the
    eval-cadence step math correct.
    """
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        logger.info("[bs-probe] no CUDA -> keeping per_device_train_batch_size=%d",
                    cfg.per_device_train_batch_size)
        return

    model = bundle.model
    dev = next(model.parameters()).device
    if dev.type != "cuda":                       # build_model leaves it on CPU; Trainer would move it
        model = model.to("cuda")
        dev = next(model.parameters()).device
    target_eff = max(1, cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps)
    logger.info("[bs-probe] start: micro-batch <= %d, max_in=%d max_out=%d, gc=%s, target effective=%d",
                cfg.per_device_train_batch_size, cfg.max_input_length, cfg.max_target_length,
                cfg.gradient_checkpointing, target_eff)

    # The probe runs the plain CE forward, NOT the custom TALLIPTrainer loss.
    # The TLC differentiable length loss materialises an extra (B, T, vocab)
    # fp32 tensor (~4 GB at B=32) that the probe cannot see -> cap the start
    # micro-batch for length-loss configs so they are not sent to a bs that
    # thrashes the allocator in real training.
    start_bs = int(cfg.per_device_train_batch_size)
    if getattr(cfg, "use_length_loss", False) and start_bs > 16:
        logger.info("[bs-probe] length-loss config: capping start micro-batch 32 -> 16 "
                    "(probe cannot model the TLC length-loss memory)")
        start_bs = 16
    cands = []
    b = start_bs
    while b >= 1:
        cands.append(b)
        b //= 2

    vocab = int(getattr(model.config, "vocab_size", 32000))
    Lin, Lout = int(cfg.max_input_length), int(cfg.max_target_length)
    total = torch.cuda.get_device_properties(dev).total_memory
    chosen = None          # only set once a candidate actually completes a step
    smallest_ok = None     # smallest candidate that ran without OOM, even under headroom
    model.train()
    if cfg.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    for b in cands:
        opt = loss = ids = att = lab = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)
            ids = torch.randint(4, vocab, (b, Lin), device=dev)
            att = torch.ones_like(ids)
            lab = torch.randint(4, vocab, (b, Lout), device=dev)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-9)
            _amp_dtype = torch.bfloat16 if cfg.bf16 else torch.float16
            with torch.autocast("cuda", dtype=_amp_dtype, enabled=bool(cfg.fp16 or cfg.bf16)):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            loss.backward()
            opt.step()
            # use RESERVED, not allocated: the caching allocator's pool is what
            # actually competes for VRAM and what thrashes near the ceiling.
            peak = max(torch.cuda.max_memory_reserved(dev), torch.cuda.max_memory_allocated(dev))
            free = 1.0 - peak / total
            logger.info("[bs-probe] micro-batch %2d: reserved %.1f/%.1f GiB (%.0f%% free)",
                        b, peak / 2 ** 30, total / 2 ** 30, free * 100)
            smallest_ok = b  # completed a real step without OOM, even if under headroom
            if free >= headroom:
                chosen = b
                break
        except torch.cuda.OutOfMemoryError:
            logger.info("[bs-probe] micro-batch %2d: OOM -> halving", b)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.info("[bs-probe] micro-batch %2d: OOM -> halving", b)
            else:
                logger.warning("[bs-probe] micro-batch %2d: non-OOM error, aborting probe (%s)", b, e)
                chosen = min(b, int(cfg.per_device_train_batch_size))
                _abort = True
        finally:
            del opt, loss, ids, att, lab
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        if locals().get("_abort"):
            break

    if chosen is None:
        if smallest_ok is not None:
            # nothing met the headroom margin, but this candidate DID complete a
            # real step without OOM -- safer bet than silently assuming 1 works.
            logger.warning("[bs-probe] no candidate reached %.0f%% headroom; falling back to "
                            "micro-batch %d (smallest that actually ran)", headroom * 100, smallest_ok)
            chosen = smallest_ok
        else:
            # every candidate down to and including micro-batch 1 hit a hard OOM:
            # this (config, max_input_length, precision) combination does not fit
            # on this GPU at all.  Fail loudly and immediately instead of quietly
            # "choosing" the unverified starting default -- that used to let
            # training proceed at a batch size that had ALSO just OOM'd, only to
            # die later inside accelerate's own find_executable_batch_size with a
            # more confusing "reached zero" error (seen on mbart50 X-1111, RoPE+
            # ALiBi+TLC stacked -- 2026-09-29).
            raise DoesNotFitError(
                f"[bs-probe] {cfg.case_id}: even micro-batch 1 (max_in={Lin}, max_out={Lout}) "
                "hits CUDA OOM -- this config does not fit on this GPU at this precision. "
                "Needs a real memory reduction (e.g. a shorter max_input_length for this "
                "factor combination) or a bigger GPU, not another retry."
            )
    ga = max(1, round(target_eff / chosen))
    logger.info("[bs-probe] -> per_device_train_batch_size=%d  gradient_accumulation_steps=%d  "
                "(effective %d; was %d x %d)", chosen, ga, chosen * ga,
                cfg.per_device_train_batch_size, cfg.gradient_accumulation_steps)
    cfg.per_device_train_batch_size = int(chosen)
    cfg.gradient_accumulation_steps = int(ga)


def run_training(
    cfg: RunConfig,
    resume: bool = False,
    skip_completed: bool = False,
    run_final_eval: bool = True,
    eval_metrics=None,
    eval_corpora: Optional[list] = None,
    eval_variants: Optional[list] = None,
    run_comet: bool = True,
    device: str = "cuda",
    train_pairs_override: Optional[tuple] = None,
    verify_dataset: bool = True,
    manifest_extra: Optional[dict] = None,
) -> dict:
    """
    train_pairs_override : (src_list, tgt_list) to use instead of the manifest's
        train files — for analyses that construct the training set on the fly
        (e.g. the V1+a*V2 scale ablation).  val/test still come from the manifest.
    verify_dataset : skip the train/val/test fingerprint check (set False when
        train is intentionally reconstructed by the caller).
    """
    run_dir = ensure_dir(cfg.output_dir)

    if skip_completed and is_completed(run_dir):
        logger.info("[skip] %s already completed", run_dir)
        return {"status": "skipped", "run_dir": run_dir}

    # frozen config next to the run
    cfg.save_yaml(os.path.join(run_dir, "config.yaml"))

    # ---- strict seeding + banner ------------------------------------
    seed_report = set_global_seed(cfg.seed, deterministic=cfg.deterministic)
    _print_banner(cfg)

    # ---- dataset manifest + fingerprint ---------------------------
    ds_manifest = _load_dataset_manifest(cfg.dataset_version)
    if verify_dataset:
        verify_fingerprint(ds_manifest, strict=True)
    else:
        logger.info("[train_run] dataset fingerprint check skipped (train set reconstructed by caller)")
    cfg.dataset_fingerprint = ds_manifest.get("fingerprint")

    # ---- build model early (so the manifest can record it) --------
    bundle = build_model(cfg)

    # ---- per-model micro-batch autoscale -------------------------
    # Start from cfg.per_device_train_batch_size (32) and halve on real CUDA OOM
    # until a full fwd+bwd+AdamW step fits with headroom; rescale
    # gradient_accumulation_steps so the EFFECTIVE batch (bs*ga) is unchanged.
    # -> each config trains at the largest micro-batch its GPU allows (faster),
    #    without near-OOM allocator thrashing, and with identical optimisation.
    if getattr(cfg, "auto_find_batch_size", False):
        try:
            _autoscale_micro_batch(bundle, cfg)
        except DoesNotFitError:
            raise  # a real capacity problem -- let this run fail loudly, not retry blind
        except Exception as exc:  # never let an unrelated probe bug kill a run
            logger.warning("[bs-probe] failed, keeping YAML batch settings (%s)", exc)
        cfg.save_yaml(os.path.join(run_dir, "config.yaml"))  # refreeze with final bs/ga

    _extra = {"seed_report": seed_report.to_dict(), "train_repeat_factor": cfg.train_repeat_factor,
              "dataset_fingerprint_verified": verify_dataset}
    if manifest_extra:
        _extra.update(manifest_extra)
    mani = run_manifest.build(cfg, bundle=bundle, dataset_manifest=ds_manifest, extra=_extra)
    mani["start_time"] = now_iso()
    run_manifest.write(run_dir, mani)

    tracker = StatusTracker(run_dir, meta={"case_id": cfg.case_id, "seed": cfg.seed})

    with tracker:
        if train_pairs_override is not None:
            train_src, train_tgt = list(train_pairs_override[0]), list(train_pairs_override[1])
            logger.info("[train_run] using caller-supplied training set: %d pairs", len(train_src))
        else:
            train_src, train_tgt = load_parallel_corpus(cfg.data_dir, cfg.train_src, cfg.train_tgt)

        if cfg.max_train_samples and len(train_src) > cfg.max_train_samples:
            if cfg.stratified_train_subsample:
                from src.data.dataset import stratified_subsample

                train_src, train_tgt = stratified_subsample(
                    train_src, train_tgt, cfg.max_train_samples, cfg.bucket_boundaries, cfg.seed
                )
                logger.info("[train_run] stratified subsample -> %d pairs "
                            "(per-bucket proportions preserved)", len(train_src))
            else:
                train_src = train_src[: cfg.max_train_samples]
                train_tgt = train_tgt[: cfg.max_train_samples]
                logger.info("[train_run] first-N subsample -> %d pairs", len(train_src))
        n_train_base = len(train_src)
        k = max(1, int(cfg.train_repeat_factor))
        if k > 1:
            train_src, train_tgt = train_src * k, train_tgt * k
            logger.info("[train_run] train_repeat_factor=%d -> %d effective pairs", k, len(train_src))

        val_src, val_tgt = load_parallel_corpus(
            cfg.data_dir, cfg.val_src, cfg.val_tgt, max_samples=cfg.max_val_samples
        )
        tracker.update_meta(n_train_base=n_train_base, n_train_effective=len(train_src),
                            train_repeat_factor=k, n_val=len(val_src))

        from src.data.dataset import SaHiDataset

        val_ds = SaHiDataset(
            val_src, val_tgt, bundle.tokenizer,
            model_family=bundle.family,
            tok_src_lang=bundle.tok_src_lang, tok_tgt_lang=bundle.tok_tgt_lang,
            max_input_length=cfg.max_input_length, max_target_length=cfg.max_target_length,
            use_length_control_token=cfg.use_length_control_token,
            bucket_boundaries=cfg.bucket_boundaries,
            bucket_mode=getattr(cfg, "bucket_mode", "source_length"),
            target_length_ratio=getattr(cfg, "target_length_ratio", 1.8017),
            predicted_target_boundaries=getattr(cfg, "predicted_target_boundaries", None),
            oracle_ratio_boundaries=getattr(cfg, "oracle_ratio_boundaries", None),
        )

        # optimiser steps per epoch (mirrors HF Trainer's own math, drop_last=False,
        # one process per GPU) — lets build_training_args honour eval_every_n_epochs.
        import math as _math
        import torch as _torch

        _ngpu = max(1, _torch.cuda.device_count())
        _steps_per_epoch = _math.ceil(
            _math.ceil(len(train_src) / (cfg.per_device_train_batch_size * _ngpu))
            / max(1, cfg.gradient_accumulation_steps)
        )
        args = build_training_args(cfg, steps_per_epoch=_steps_per_epoch)
        compute_metrics = make_compute_metrics(
            bundle.tokenizer, sources=val_src, comet_every=cfg.comet_eval_every, run_comet=run_comet
        )

        from transformers import DataCollatorForSeq2Seq

        eval_base = DataCollatorForSeq2Seq(tokenizer=bundle.tokenizer, model=bundle.model, padding=True)

        def eval_collator(features):
            for feat in features:
                feat.pop("bucket", None)
                feat.pop("target_length", None)
            return eval_base(features)

        trainer = TALLIPTrainer(
            cfg=cfg,
            bundle=bundle,
            train_src=train_src,
            train_tgt=train_tgt,
            model=bundle.model,
            args=args,
            eval_dataset=val_ds,
            data_collator=eval_collator,
            compute_metrics=compute_metrics,
        )
        if cfg.early_stopping_patience:
            from transformers import EarlyStoppingCallback

            trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience))

        resume_ckpt = None
        if resume:
            resume_ckpt = _latest_checkpoint(os.path.join(run_dir, "checkpoint"))
            if resume_ckpt:
                logger.info("[resume] from %s", resume_ckpt)

        trainer.train(resume_from_checkpoint=resume_ckpt)

        best_dir = os.path.join(run_dir, "checkpoint-best")
        trainer.save_model(best_dir)
        bundle.tokenizer.save_pretrained(best_dir)
        logger.info("[train_run] best checkpoint -> %s", best_dir)

        if getattr(cfg, "bucket_embed_dim", None):
            # the trained BucketAwareEmbedding's weights (base/bucket_embed/proj)
            # got saved above, but a plain AutoModelForSeq2SeqLM.from_pretrained
            # reload (every downstream eval script, including the one 20 lines
            # below) builds the STOCK architecture and silently drops them as
            # "unused" -- write a sidecar so _load_ckpt knows to re-apply the
            # wrapper and reload those specific weights. See src/models/bucket_embedding.py.
            from src.models.builder import LENGTH_CONTROL_TOKENS
            from src.utils.io import write_json

            bucket_ids = [bundle.tokenizer.convert_tokens_to_ids(t) for t in LENGTH_CONTROL_TOKENS]
            write_json(os.path.join(best_dir, "bucket_embed_dim.json"),
                      {"bucket_embed_dim": int(cfg.bucket_embed_dim), "bucket_token_ids": bucket_ids})

        test_payload = None
        corpora = list(eval_corpora) if eval_corpora else ["v2"]
        variants = list(eval_variants) if eval_variants else ["unsampled"]
        if run_final_eval:
            logger.info("[train_run] final held-out evaluation — corpora=%s x variants=%s "
                        "(config frozen)", corpora, variants)
            variant_payloads = evaluate_test_variants(
                cfg,
                checkpoint_path=best_dir,
                corpora=corpora,
                variants=variants,
                num_beams=cfg.final_eval_num_beams,
                max_samples=cfg.max_test_samples,
                metrics=eval_metrics,
                run_comet=run_comet,
                device=device,
            )
            # headline = own training corpus, first requested variant
            headline_variant = variants[0]
            test_payload = variant_payloads.get((cfg.dataset_version, headline_variant)) \
                or next(iter(variant_payloads.values()), None)

        run_manifest.stamp_times(run_dir, end=now_iso())

        eval_subpath = os.path.join("evaluation", cfg.dataset_version,
                                    variants[0] if variants else "unsampled")
        problems = check_run(run_dir, expected_min_steps=1,
                             split="val" if not run_final_eval else eval_subpath,
                             require_eval=run_final_eval)
        if problems:
            tracker.set("failed", failure_reason="; ".join(problems))
            logger.error("[train_run] FAILED validation: %s", problems)
            return {"status": "failed", "run_dir": run_dir, "problems": problems}

        tracker.set("completed")

    summary = {"status": "completed", "run_dir": run_dir}
    if test_payload:
        summary["test_metrics"] = {
            k: test_payload["metrics"].get(k)
            for k in ("bleu", "chrf2", "ter", "meteor", "bertscore", "comet")
        }
    return summary
