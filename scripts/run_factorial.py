#!/usr/bin/env python
"""
Main 3-seed 2^4 factorial runner (brief sections 3, 29-A, 37-P1).

    16 X-ABCD configs  x  seeds {42,43,44}  x  dataset v2  = 48 runs
    (v2 ONLY — never a full factorial on multiple dataset versions).

Examples
--------
  # one run
  python scripts/run_factorial.py --config X-1101 --seed 42 --dataset v2

  # everything (sequential; use scripts/generate_jobs.py for multi-GPU)
  python scripts/run_factorial.py --run-all --dataset v2 --seeds 42 43 44

  # resume interrupted + skip finished
  python scripts/run_factorial.py --run-all --resume --skip-completed

  # tiny smoke test
  python scripts/run_factorial.py --config X-0000 --seed 42 --epochs 1 \\
      --max-train-samples 200 --max-val-samples 64 --max-test-samples 64
"""
import _paths  # noqa: F401

import argparse
import logging
import os
import sys

from src.config import ALL_CASE_IDS, RunConfig, case_label, load_experiment_yaml, parse_case_id
from src.data.dataset import normalized_bucket_weights
from src.experiments.factorial import enumerate_runs
from src.experiments.status import read_status
from src.experiments.train_run import run_training
from src.utils.io import read_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("run_factorial")


MANUAL_BBS = [1.0, 1.0, 1.2, 1.5, 2.0]


# each architecture gets its OWN frozen BBS search (they don't share a folder,
# so running one architecture's search never clobbers another's selected vector)
_BBS_SEARCH_DIR = {"nllb": "bbs_search", "mbart50_many_to_many": "bbs_search_mbart"}


def _resolve_bbs_weights(args, dataset, experiments_root, run_group, model_key="nllb"):
    """
    Returns (weights_list, source_str).  Priority-3 -> Priority-1 workflow:
    the grid search runs FIRST and freezes selected_bbs_weights.json; the
    factorial then trains with that vector.

      --bbs-weights selected : require selected_bbs_weights.json (error if absent)
      --bbs-weights auto     : use it if present, else fall back to manual (warn)
      --bbs-weights manual   : the current-paper vector [1,1,1.2,1.5,2.0]
      --bbs-weights "a b c d e" : an explicit vector
    All are mean-normalised (sum=5) before use.
    """
    mode = args.bbs_weights
    bbs_dir = _BBS_SEARCH_DIR.get(model_key, f"bbs_search_{model_key}")
    sel_path = os.path.join(experiments_root, dataset, bbs_dir, "selected_bbs_weights.json")

    if mode == "manual":
        return normalized_bucket_weights(MANUAL_BBS), "manual [1,1,1.2,1.5,2.0]"
    if " " in mode or "," in mode:
        vec = [float(x) for x in mode.replace(",", " ").split()]
        if len(vec) != 5:
            raise SystemExit(f"--bbs-weights needs 5 numbers, got {vec}")
        return normalized_bucket_weights(vec), f"explicit {vec}"

    # auto / selected
    if os.path.isfile(sel_path):
        j = read_json(sel_path)
        vec = j.get("selected_vector_normalized") or normalized_bucket_weights(j["selected_vector_raw"])
        vec = normalized_bucket_weights(vec)
        src = (f"grid-selected ({j.get('candidate_id')}, {j.get('selection_metric')}="
               f"{j.get('validation_score')}, downstream={j.get('downstream_config')})")
        return vec, src
    if mode == "selected":
        raise SystemExit(
            f"--bbs-weights selected but {sel_path} is missing.\n"
            "Run the BBS grid search first:\n"
            f"  python scripts/run_bbs_search.py --dataset {dataset} --config X-1000 --selection-metric comet\n"
            "or pass --bbs-weights manual / auto.")
    log.warning("[bbs] no %s yet — falling back to the MANUAL vector. "
                "Run scripts/run_bbs_search.py first for the grid-selected weights.", sel_path)
    return normalized_bucket_weights(MANUAL_BBS), "manual (fallback; grid search not run yet)"


def _overrides(args) -> dict:
    o = {}
    if args.epochs is not None:
        o["num_train_epochs"] = args.epochs
    if args.batch_size is not None:
        o["per_device_train_batch_size"] = args.batch_size
    if args.lr is not None:
        o["learning_rate"] = args.lr
    if args.max_train_samples is not None:
        o["max_train_samples"] = args.max_train_samples
    if args.max_val_samples is not None:
        o["max_val_samples"] = args.max_val_samples
    if args.max_test_samples is not None:
        o["max_test_samples"] = args.max_test_samples
    if args.length_loss_lambda is not None:
        o["length_loss_lambda"] = args.length_loss_lambda
    if args.length_loss_type is not None:
        o["length_loss_type"] = args.length_loss_type
    if args.bucket_embed_dim is not None:
        o["bucket_embed_dim"] = args.bucket_embed_dim
    if args.bucket_mode is not None:
        o["bucket_mode"] = args.bucket_mode
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="single X-ABCD case id")
    ap.add_argument("--seed", type=int, help="single seed")
    ap.add_argument("--run-all", action="store_true", help="all configs x all seeds")
    ap.add_argument("--configs", nargs="*", help="explicit config list (overrides YAML)")
    ap.add_argument("--seeds", nargs="*", type=int, help="explicit seed list (overrides YAML)")
    ap.add_argument("--dataset", default="v2")
    ap.add_argument("--model", default=None, help="model key (default from experiment YAML: nllb)")
    ap.add_argument("--experiment", default="factorial", help="configs/experiments/<name>.yaml")
    ap.add_argument("--experiments-root", default="experiments")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-completed", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument("--list", action="store_true", help="print the 16 configs and exit")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-final-eval", action="store_true", help="skip the held-out test eval")
    ap.add_argument("--eval-corpora", nargs="*", default=None,
                    help="override the experiment YAML's evaluation.eval_corpora "
                         "(e.g. --eval-corpora v1 to score a v1-trained run on v1's own "
                         "test set instead of the YAML default)")
    ap.add_argument("--bbs-weights", default="auto",
                    help="BBS weight vector for A-bit=1 configs: "
                         "auto (grid-selected if available, else manual) | selected | manual | \"w1 w2 w3 w4 w5\"")
    # smoke-test / override knobs
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--max-train-samples", type=int)
    ap.add_argument("--max-val-samples", type=int)
    ap.add_argument("--max-test-samples", type=int)
    ap.add_argument("--no-comet", action="store_true")
    ap.add_argument("--run-group", default=None, help="override the experiment YAML's run_group "
                    "(use to isolate ablation variants under their own output path)")
    ap.add_argument("--length-loss-lambda", type=float, default=None,
                    help="override RunConfig.length_loss_lambda")
    ap.add_argument("--length-loss-type", default=None,
                    choices=["l1", "squared", "huber", "l1_minus1"],
                    help="override RunConfig.length_loss_type (see src/models/length_loss.py)")
    ap.add_argument("--bucket-embed-dim", type=int, default=None,
                    help="override RunConfig.bucket_embed_dim (dedicated embedding+projection dim "
                         "for the 5 LEN_BUCKET tokens, independent of d_model; default: tied vocab "
                         "embedding, unchanged)")
    ap.add_argument("--bucket-mode", default=None,
                    choices=["source_length", "predicted_target_length", "oracle_ratio_class"],
                    help="override RunConfig.bucket_mode -- which value the 5 TLC buckets are "
                         "computed from (see src/data/dataset.py::SaHiDataset)")
    args = ap.parse_args()

    if args.list:
        print("X-ABCD  (A=BBS  B=TLC  C=RoPE  D=ALiBi)")
        for cid in ALL_CASE_IDS:
            f = parse_case_id(cid)
            print(f"  {cid}   A={f['bbs']} B={f['tlc']} C={f['rope']} D={f['alibi']}   {case_label(cid)}")
        return

    exp_yaml = load_experiment_yaml(args.experiment)
    model_key = args.model or exp_yaml.get("model_key", "nllb")
    run_group = args.run_group or exp_yaml.get("run_group", args.experiment)
    _ev = exp_yaml.get("evaluation") or {}
    eval_metrics = _ev.get("metrics")
    eval_corpora = args.eval_corpora if args.eval_corpora else _ev.get("eval_corpora", ["v2"])
    eval_variants = _ev.get("test_variants", ["unsampled"])

    # ---- resolve the run list --------------------------------------
    if args.config and args.seed is not None:
        runs = [(args.config.upper(), args.seed)]
    elif args.run_all or args.configs or args.seeds:
        cfgs = args.configs or ("all" if exp_yaml.get("configs") in (None, "all") else exp_yaml["configs"])
        sds = args.seeds or exp_yaml.get("seeds", [42, 43, 44])
        runs = enumerate_runs(cfgs, sds)
    else:
        ap.error("give --config/--seed, or --run-all, or --configs/--seeds")

    overrides = _overrides(args)
    bbs_vec, bbs_src = _resolve_bbs_weights(args, args.dataset, args.experiments_root, run_group, model_key)
    overrides["bucket_weights"] = bbs_vec
    print(f"\n{'='*70}\n  PLAN — {len(runs)} run(s)   experiment={args.experiment}  "
          f"dataset={args.dataset}  model={model_key}\n"
          f"  BBS weights (A-bit=1 configs): {[round(x,3) for x in bbs_vec]}   source: {bbs_src}\n{'='*70}")
    for i, (cid, seed) in enumerate(runs, 1):
        cfg = RunConfig.resolve(cid, seed, args.dataset, model_key, args.experiment,
                                run_group=run_group, experiments_root=args.experiments_root,
                                overrides=overrides)
        st = read_status(cfg.output_dir)
        tag = f"[{st['status']}]" if st else "[new]"
        print(f"  {i:>3}. {cid} seed={seed:<4} {tag:<14} -> {cfg.output_dir}")
    if overrides:
        print(f"\n  overrides: {overrides}")
    print(f"  resume={args.resume}  skip_completed={args.skip_completed}\n{'='*70}\n")

    if args.dry_run:
        print("dry run — nothing executed.")
        return

    results = []
    for i, (cid, seed) in enumerate(runs, 1):
        log.info("[%d/%d] %s seed=%d", i, len(runs), cid, seed)
        cfg = RunConfig.resolve(cid, seed, args.dataset, model_key, args.experiment,
                                run_group=run_group, experiments_root=args.experiments_root,
                                overrides=overrides)
        try:
            res = run_training(cfg, resume=args.resume, skip_completed=args.skip_completed,
                               run_final_eval=not args.no_final_eval, eval_metrics=eval_metrics,
                               eval_corpora=eval_corpora, eval_variants=eval_variants, run_comet=not args.no_comet,
                               device=args.device,
                               manifest_extra={"bbs_weights_source": bbs_src,
                                               "bucket_weights_normalized": bbs_vec})
        except Exception as exc:  # keep the batch alive
            log.exception("run failed: %s", exc)
            res = {"status": "failed", "run_dir": cfg.output_dir, "error": str(exc)}
        res["config"], res["seed"] = cid, seed
        results.append(res)

    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    for r in results:
        tm = r.get("test_metrics", {})
        extra = f"  BLEU={tm.get('bleu'):.2f} COMET={tm.get('comet')}" if tm.get("bleu") is not None else ""
        print(f"  {r['config']} seed={r['seed']:<4} {r['status']:<10}{extra}")
    n_ok = sum(1 for r in results if r["status"] == "completed")
    print(f"\n  completed={n_ok}  other={len(results)-n_ok}")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
