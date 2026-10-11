"""
Seq2Seq trainer for TLC_TALLIP.

Wraps HF `Seq2SeqTrainer` with:
  * a seeded bucket-balanced sampler (A = BBS) — deterministic per seed;
  * the differentiable TLC length loss (B = TLC), switchable independently of
    the length-control token;
  * a training-history dump to ``<run_dir>/metrics.json`` after every eval.
"""
from __future__ import annotations

import inspect
import logging
import os
from typing import List, Optional

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback

from src.config import RunConfig
from src.data.dataset import SaHiDataset, build_weighted_sampler
from src.models.length_loss import LengthAwareLoss
from src.utils.io import write_json
from src.utils.seeding import make_generator, seed_worker

logger = logging.getLogger(__name__)

_NON_MODEL_KEYS = ("bucket", "target_length")


class TALLIPTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        cfg: RunConfig,
        bundle,
        train_src: List[str],
        train_tgt: List[str],
        *args,
        **kwargs,
    ):
        self.cfg = cfg
        self.bundle = bundle
        self._tok = bundle.tokenizer
        self.train_src = train_src
        self.train_tgt = train_tgt

        self.length_loss = None
        if cfg.use_length_loss:
            self.length_loss = LengthAwareLoss(
                eos_id=bundle.eos_token_id,
                length_loss_lambda=cfg.length_loss_lambda,
                length_loss_epsilon=cfg.length_loss_epsilon,
                label_smoothing=cfg.label_smoothing,
                use_length_loss=True,
                loss_type=getattr(cfg, "length_loss_type", "l1"),
            )
        super().__init__(*args, **kwargs)
        self.add_callback(_HistoryDump(cfg.output_dir))

    # ---- loss ---------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        target_length = inputs.pop("target_length", None)
        inputs.pop("bucket", None)
        if self.length_loss is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kw)
        labels = inputs["labels"]
        outputs = model(**inputs)
        loss = self.length_loss(outputs.logits, labels, target_length)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        for k in _NON_MODEL_KEYS:
            inputs.pop(k, None)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

    # ---- data -------------------------------------------------------
    def _collator(self):
        base = DataCollatorForSeq2Seq(tokenizer=self._tok, model=self.model, padding=True)

        def collate(features):
            tl = [f.pop("target_length", 0) for f in features]
            for f in features:
                f.pop("bucket", None)
            batch = base(features)
            batch["target_length"] = torch.tensor(tl, dtype=torch.long)
            return batch

        return collate

    def get_train_dataloader(self) -> DataLoader:
        ds = SaHiDataset(
            self.train_src,
            self.train_tgt,
            self._tok,
            model_family=self.bundle.family,
            tok_src_lang=self.bundle.tok_src_lang,
            tok_tgt_lang=self.bundle.tok_tgt_lang,
            max_input_length=self.cfg.max_input_length,
            max_target_length=self.cfg.max_target_length,
            use_length_control_token=self.cfg.use_length_control_token,
            bucket_boundaries=self.cfg.bucket_boundaries,
            bucket_mode=getattr(self.cfg, "bucket_mode", "source_length"),
            target_length_ratio=getattr(self.cfg, "target_length_ratio", 1.8017),
            predicted_target_boundaries=getattr(self.cfg, "predicted_target_boundaries", None),
            oracle_ratio_boundaries=getattr(self.cfg, "oracle_ratio_boundaries", None),
        )
        common = dict(
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self._collator(),
            num_workers=self.cfg.dataloader_num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            drop_last=False,
        )
        if self.cfg.use_bucket_balanced_sampling:
            sampler = build_weighted_sampler(
                ds, self.cfg.bucket_weights, seed=self.cfg.seed, deterministic=self.cfg.bbs_deterministic
            )
            return DataLoader(ds, sampler=sampler, **common)
        return DataLoader(ds, shuffle=True, generator=make_generator(self.cfg.seed), **common)


class _HistoryDump(TrainerCallback):
    def __init__(self, run_dir: str):
        self.path = os.path.join(run_dir, "metrics.json")

    def _dump(self, state):
        write_json(
            self.path,
            {
                "global_step": state.global_step,
                "epoch": state.epoch,
                "best_metric": state.best_metric,
                "best_model_checkpoint": state.best_model_checkpoint,
                "log_history": state.log_history,
            },
        )

    def on_evaluate(self, args, state, control, **kw):
        self._dump(state)

    def on_train_end(self, args, state, control, **kw):
        self._dump(state)


# ---------------------------------------------------------------------------
def build_training_args(cfg: RunConfig, steps_per_epoch: Optional[int] = None) -> Seq2SeqTrainingArguments:
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    kw = {}

    # eval / checkpoint cadence.  Default: whatever cfg.eval_strategy says.
    # If eval_every_n_epochs > 1 (and we know steps/epoch), switch to a 'steps'
    # schedule firing every N epochs so validation is not run after every epoch.
    eff_eval_strategy = cfg.eval_strategy
    eff_save_strategy = cfg.save_strategy
    every_n = int(getattr(cfg, "eval_every_n_epochs", 1) or 1)
    if every_n > 1 and steps_per_epoch and cfg.eval_strategy == "epoch":
        stride = max(1, every_n * int(steps_per_epoch))
        eff_eval_strategy = "steps"
        kw["eval_steps"] = stride
        if cfg.save_strategy == "epoch":
            eff_save_strategy = "steps"
            kw["save_steps"] = stride

    if "eval_strategy" in sig.parameters:
        kw["eval_strategy"] = eff_eval_strategy
    else:
        kw["evaluation_strategy"] = eff_eval_strategy

    if "generation_max_new_tokens" in sig.parameters:
        kw["generation_max_new_tokens"] = cfg.max_target_length
    else:
        kw["generation_max_length"] = cfg.max_target_length
    if "generation_num_beams" in sig.parameters:
        kw["generation_num_beams"] = cfg.train_eval_num_beams
    if "save_safetensors" in sig.parameters:
        kw["save_safetensors"] = False
    if "torch_compile" in sig.parameters:
        kw["torch_compile"] = False
    if "data_seed" in sig.parameters:
        kw["data_seed"] = cfg.seed
    if "dataloader_drop_last" in sig.parameters:
        kw["dataloader_drop_last"] = False

    # start at per_device_train_batch_size and halve on CUDA OOM until it fits
    if getattr(cfg, "auto_find_batch_size", False) and "auto_find_batch_size" in sig.parameters:
        kw["auto_find_batch_size"] = True

    metric = cfg.metric_for_best_model
    greater = cfg.greater_is_better

    return Seq2SeqTrainingArguments(
        output_dir=os.path.join(cfg.output_dir, "checkpoint"),
        overwrite_output_dir=False,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_checkpointing=cfg.gradient_checkpointing,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        lr_scheduler_type=cfg.lr_scheduler_type,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        optim=cfg.optimizer,
        save_strategy=eff_save_strategy,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=metric,
        greater_is_better=greater,
        logging_steps=cfg.logging_steps,
        predict_with_generate=True,
        report_to=["none"],
        dataloader_num_workers=cfg.dataloader_num_workers,
        seed=cfg.seed,
        run_name=f"{cfg.case_id}_seed{cfg.seed}",
        ddp_find_unused_parameters=False,
        **kw,
    )
