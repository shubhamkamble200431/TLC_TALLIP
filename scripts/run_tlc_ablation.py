#!/usr/bin/env python
"""
TLC length-loss lambda sensitivity + optional token/loss decomposition
(brief sections 19-21, 29-E/F, 37-P9).

VALIDATION-ONLY selection ("Never tune lambda on the test set").

Outputs (under experiments/<dataset>/tlc_ablation/):
    tlc_lambda_ablation.csv       lambda, BLEU, chrF2, COMET, length_error, B5_BLEU
    tlc_decomposition.csv         (optional) token-only / loss-only / both / neither
    selected_lambda.json

Examples
--------
  python scripts/run_tlc_ablation.py --dataset v2
  python scripts/run_tlc_ablation.py --epochs 1 --max-train-samples 200 --lambdas 0.0 0.1
"""
import _paths  # noqa: F401

import argparse
import csv
import logging
import os
import sys

from src.config import RunConfig, load_experiment_yaml
from src.data.buckets import BUCKET_LABELS_LONG
from src.evaluation.evaluate import evaluate_checkpoint
from src.experiments.train_run import run_training
from src.utils.io import ensure_dir, read_json, write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("run_tlc_ablation")


def _val_eval(cfg, best_dir, device, run_comet):
    payload = evaluate_checkpoint(cfg, checkpoint_path=best_dir, split="val",
                                  num_beams=cfg.final_eval_num_beams, run_comet=run_comet,
                                  device=device, max_samples=cfg.max_val_samples,
                                  metrics=["bleu", "chrf2", "comet"])
    m = payload["metrics"]
    b5 = BUCKET_LABELS_LONG[5]
    return {
        "bleu": m.get("bleu"), "chrf2": m.get("chrf2"), "comet": m.get("comet"),
        "length_error": m.get("absolute_length_error"),
        "b5_bleu": m.get(f"{b5}_bleu"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="v2")
    ap.add_argument("--experiment", default="tlc_lambda")
    ap.add_argument("--experiments-root", default="experiments")
    ap.add_argument("--lambdas", nargs="*", type=float)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--decomposition", action="store_true", help="also run token/loss decomposition")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-completed", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--max-train-samples", type=int)
    ap.add_argument("--max-val-samples", type=int)
    ap.add_argument("--no-comet", action="store_true")
    args = ap.parse_args()

    exp = load_experiment_yaml(args.experiment)
    base_cfg = exp.get("base_config", "X-0100")
    lambdas = args.lambdas or exp.get("lambda_grid", [0.0, 0.01, 0.05, 0.1, 0.2, 0.5])
    seed = args.seed or (exp.get("seeds", [42]) or [42])[0]
    run_group = exp.get("run_group", "tlc_ablation")
    model_key = exp.get("model_key", "nllb")
    run_comet = not args.no_comet

    overrides = dict(exp.get("defaults") or {})
    if args.epochs is not None:
        overrides["num_train_epochs"] = args.epochs
    if args.max_train_samples is not None:
        overrides["max_train_samples"] = args.max_train_samples
    if args.max_val_samples is not None:
        overrides["max_val_samples"] = args.max_val_samples

    out_root = ensure_dir(os.path.join(args.experiments_root, args.dataset, run_group))
    lam_csv = os.path.join(out_root, "tlc_lambda_ablation.csv")
    sel_json = os.path.join(out_root, "selected_lambda.json")

    print(f"[tlc_ablation] base={base_cfg} seed={seed} lambdas={lambdas}")
    if args.dry_run:
        print("dry run — nothing executed.")
        return

    rows = []
    for lam in lambdas:
        ov = {**overrides, "length_loss_lambda": lam, "use_length_loss": lam > 0.0,
              "use_length_control_token": True}
        cfg = RunConfig.resolve(base_cfg, seed, args.dataset, model_key, args.experiment,
                                run_group=os.path.join(run_group, f"lambda_{lam}"),
                                experiments_root=args.experiments_root, overrides=ov)
        try:
            run_training(cfg, skip_completed=args.skip_completed, run_final_eval=False, device=args.device)
            vs = _val_eval(cfg, os.path.join(cfg.output_dir, "checkpoint-best"), args.device, run_comet)
            status = "done"
        except Exception as exc:
            log.exception("lambda %s failed", lam)
            vs = {k: None for k in ("bleu", "chrf2", "comet", "length_error", "b5_bleu")}
            status = f"error: {exc}"
        rows.append({"lambda": lam, "seed": seed, "status": status, **vs})
        with open(lam_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["lambda", "seed", "status", "bleu", "chrf2",
                                              "comet", "length_error", "b5_bleu"])
            w.writeheader()
            w.writerows(rows)
        log.info("[tlc_ablation] lambda=%s  BLEU=%s COMET=%s len_err=%s", lam,
                 vs["bleu"], vs["comet"], vs["length_error"])

    valid = [r for r in rows if r["comet"] is not None]
    if valid:
        best = max(valid, key=lambda r: r["comet"])
        write_json(sel_json, {
            "selected_lambda": best["lambda"], "selection_metric": "val_comet",
            "validation_score": best["comet"], "seed": seed, "base_config": base_cfg,
            "grid": lambdas, "results_csv": lam_csv, "test_set_used": False,
        })
        print(f"[tlc_ablation] SELECTED lambda={best['lambda']}  val_comet={best['comet']}")

    # ---- optional token / loss decomposition (section 19) ---------
    if args.decomposition or (exp.get("decomposition") or {}).get("enabled"):
        dec_csv = os.path.join(out_root, "tlc_decomposition.csv")
        variants = (exp.get("decomposition") or {}).get("variants") or [
            {"use_length_control_token": False, "use_length_loss": False},
            {"use_length_control_token": True, "use_length_loss": False},
            {"use_length_control_token": False, "use_length_loss": True},
            {"use_length_control_token": True, "use_length_loss": True},
        ]
        drows = []
        for v in variants:
            name = f"tok{int(v['use_length_control_token'])}_loss{int(v['use_length_loss'])}"
            ov = {**overrides, **v, "length_loss_lambda": overrides.get("length_loss_lambda", 0.1)}
            cfg = RunConfig.resolve(base_cfg, seed, args.dataset, model_key, args.experiment,
                                    run_group=os.path.join(run_group, "decomp", name),
                                    experiments_root=args.experiments_root, overrides=ov)
            try:
                run_training(cfg, skip_completed=args.skip_completed, run_final_eval=False, device=args.device)
                vs = _val_eval(cfg, os.path.join(cfg.output_dir, "checkpoint-best"), args.device, run_comet)
                status = "done"
            except Exception as exc:
                vs = {k: None for k in ("bleu", "chrf2", "comet", "length_error", "b5_bleu")}
                status = f"error: {exc}"
            drows.append({"variant": name, "use_length_control_token": v["use_length_control_token"],
                          "use_length_loss": v["use_length_loss"], "seed": seed, "status": status, **vs})
            with open(dec_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(drows[0].keys()))
                w.writeheader()
                w.writerows(drows)
        print(f"[tlc_ablation] decomposition -> {dec_csv}")


if __name__ == "__main__":
    main()
