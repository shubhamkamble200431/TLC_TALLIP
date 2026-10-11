"""Experiment manifest (brief section 25) — one JSON per run."""
from __future__ import annotations

import os
from typing import Optional

from src.config import RunConfig, FACTOR_LETTERS, FACTOR_DISPLAY
from src.utils.env_info import collect as collect_env
from src.utils.io import read_json, write_json


def build(cfg: RunConfig, bundle=None, dataset_manifest: Optional[dict] = None, extra: Optional[dict] = None) -> dict:
    f = cfg.factors
    m = {
        "experiment_id": f"{cfg.dataset_version}/{cfg.run_group}/{cfg.case_id}/seed_{cfg.seed}",
        "experiment_family": cfg.experiment,
        "run_group": cfg.run_group,
        "case_id": cfg.case_id,
        "case_label": " + ".join(FACTOR_DISPLAY[k] for k in f if f[k]) or "baseline",
        "factors": {FACTOR_LETTERS[k]: f[k] for k in f},
        "factor_names": {"A": "BBS", "B": "TLC", "C": "RoPE", "D": "ALiBi"},
        "seed": cfg.seed,
        "dataset_version": cfg.dataset_version,
        "dataset_fingerprint": cfg.dataset_fingerprint,
        "model_key": cfg.model_key,
        "model_family": cfg.model_family,
        "model_name": cfg.base_model,
        "model_revision": cfg.model_revision,
        "tokenizer": cfg.base_model,
        "learning_rate": cfg.learning_rate,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "effective_batch_size": cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps,
        "num_train_epochs": cfg.num_train_epochs,
        "optimizer": cfg.optimizer,
        "lr_scheduler_type": cfg.lr_scheduler_type,
        "max_source_length": cfg.max_input_length,
        "max_target_length": cfg.max_target_length,
        "bucket_boundaries": cfg.bucket_boundaries,
        "bucket_weights_raw": cfg.bucket_weights,
        "use_bucket_balanced_sampling": cfg.use_bucket_balanced_sampling,
        "use_length_control_token": cfg.use_length_control_token,
        "use_length_loss": cfg.use_length_loss,
        "length_loss_lambda": cfg.length_loss_lambda,
        "use_rope": cfg.use_rope,
        "use_alibi": cfg.use_alibi,
        "src_lang": cfg.src_lang,
        "tgt_lang": cfg.tgt_lang,
        "mbart_src_lang": cfg.mbart_src_lang,
        "mbart_tgt_lang": cfg.mbart_tgt_lang,
        "test_set_used_for_selection": False,   # brief section 28 — invariant
        "checkpoint_selection_metric": cfg.metric_for_best_model,
        "checkpoint_selection_split": "validation",
        "environment": collect_env(),
        "start_time": None,
        "end_time": None,
    }
    if bundle is not None:
        m["forced_bos_token_id"] = bundle.forced_bos_token_id
        m["eos_token_id"] = bundle.eos_token_id
        m["rope_layers_patched"] = bundle.rope_layers
        m["alibi_layers_patched"] = bundle.alibi_layers
        m["added_length_tokens"] = bundle.added_length_tokens
    if dataset_manifest is not None:
        m["dataset_statistics"] = dataset_manifest.get("statistics", {})
        m["dataset_preprocessing"] = dataset_manifest.get("preprocessing", {})
    if extra:
        m.update(extra)
    return m


def write(run_dir: str, manifest: dict) -> str:
    path = os.path.join(run_dir, "manifest.json")
    write_json(path, manifest)
    return path


def stamp_times(run_dir: str, start=None, end=None):
    path = os.path.join(run_dir, "manifest.json")
    if not os.path.isfile(path):
        return
    m = read_json(path)
    if start:
        m["start_time"] = start
    if end:
        m["end_time"] = end
    write_json(path, m)
