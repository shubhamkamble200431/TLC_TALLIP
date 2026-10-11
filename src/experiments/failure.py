"""
Automatic failure detection (brief section 26).

`check_run(run_dir, cfg)` inspects a finished run directory and returns a list
of problems.  A non-empty list means the run must NOT be marked completed.
"""
from __future__ import annotations

import math
import os
from typing import List

from src.utils.io import read_json, read_lines


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def check_training_history(run_dir: str, expected_min_steps: int = 1) -> List[str]:
    problems = []
    mpath = os.path.join(run_dir, "metrics.json")
    if not os.path.isfile(mpath):
        return ["missing metrics.json (training history)"]
    hist = read_json(mpath)
    logs = hist.get("log_history", [])
    step_losses = [e["loss"] for e in logs if "loss" in e]
    final_losses = [e["train_loss"] for e in logs if "train_loss" in e]
    all_losses = step_losses + final_losses
    if not all_losses:
        # only a real problem if training also produced no steps at all
        if hist.get("global_step", 0) <= 0:
            problems.append("no training loss logged and global_step=0")
    if any(not _finite(l) for l in all_losses):
        problems.append("NaN/Inf training loss detected")
    # HF's fp16 GradScaler skips optimizer.step() on an overflowed gradient and
    # logs grad_norm=nan for that step, while `loss` (the forward-pass value,
    # logged separately) can still look like a normal finite number -- so the
    # loss-only check above can miss a run where EVERY step was skipped and no
    # real training happened at all (confirmed: mbart50 X-0001/seed_42,
    # 2026-09-23 -- loss ~6.0 throughout, grad_norm nan from step 50, BLEU 7.1).
    grad_norms = [e["grad_norm"] for e in logs if "grad_norm" in e]
    if grad_norms:
        bad_frac = sum(1 for g in grad_norms if not _finite(g)) / len(grad_norms)
        if bad_frac > 0.8:
            problems.append(
                f"grad_norm non-finite on {bad_frac:.0%} of logged steps "
                "(optimizer steps likely skipped throughout -- no real training happened)"
            )
    if hist.get("global_step", 0) < expected_min_steps:
        problems.append(
            f"training stopped early: global_step={hist.get('global_step')} < {expected_min_steps}"
        )
    return problems


def check_checkpoint(run_dir: str) -> List[str]:
    best = os.path.join(run_dir, "checkpoint-best")
    if not os.path.isdir(best):
        return ["missing checkpoint-best/"]
    has_weights = any(
        os.path.isfile(os.path.join(best, f))
        for f in ("pytorch_model.bin", "model.safetensors")
    ) or any(f.startswith("pytorch_model") for f in os.listdir(best))
    return [] if has_weights else ["checkpoint-best/ has no model weights"]


def check_evaluation(run_dir: str, split: str = "test") -> List[str]:
    # `split` may be a bare split name ("val") or an evaluation subpath
    # ("evaluation/v2/sampled").
    ev = os.path.join(run_dir, split) if split.startswith("evaluation") \
        else os.path.join(run_dir, "evaluation", split)
    rel = os.path.relpath(ev, run_dir)
    problems = []
    for f in ("predictions", "references", "source", "metrics.json"):
        if not os.path.isfile(os.path.join(ev, f)):
            problems.append(f"missing {rel}/{f}")
    if problems:
        return problems
    preds = read_lines(os.path.join(ev, "predictions"))
    refs = read_lines(os.path.join(ev, "references"))
    if len(preds) != len(refs):
        problems.append(f"prediction/reference count mismatch ({len(preds)} vs {len(refs)})")
    n_empty = sum(1 for p in preds if not p.strip())
    if preds and n_empty == len(preds):
        problems.append("all predictions are empty")
    if preds and n_empty / len(preds) > 0.10:
        problems.append(f"{n_empty}/{len(preds)} zero-length predictions (>10%)")
    m = read_json(os.path.join(ev, "metrics.json")).get("metrics", {})
    for key in ("bleu", "chrf2"):
        if key in m and not _finite(m[key]):
            problems.append(f"invalid {key} value: {m[key]!r}")
    if _finite(m.get("bleu")) and float(m.get("bleu", 0)) == 0.0 and len(preds) > 50:
        problems.append("BLEU is exactly 0.0 on a non-trivial test set (suspect all-EOS / broken decode)")
    return problems


def check_run(run_dir: str, expected_min_steps: int = 1, split: str = "evaluation/v2/sampled",
              require_eval: bool = True) -> List[str]:
    problems = check_training_history(run_dir, expected_min_steps) + check_checkpoint(run_dir)
    if require_eval:
        problems += check_evaluation(run_dir, split)
    return problems
