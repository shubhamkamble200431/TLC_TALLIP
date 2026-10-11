#!/usr/bin/env python
"""
V1 + alpha*V2 data-scale ablation  (repetitions x v2_fraction).

ANALYSIS ONLY — never used to tune the final test-set model.  For every
(repetitions, v2_fraction) cell it deterministically builds

    V1 + alpha*V2   (V1 = data/v1/train.* fixed ; alpha = nested prefix of a
                     shuffled data/v2/train.* ; no de-duplication)

replicates it `repetitions` times via the existing `train_repeat_factor`
mechanism, fine-tunes `downstream_config` (default X-0000), and evaluates on the
SAME held-out set / pipeline as the main experiments (V2 val + V2 test_7264_5).

Writes:
    results/csv/v1_v2_scale_ablation.csv
      repetitions, v2_fraction, training_examples, bleu, chrf2, comet,
      length_error, seed, status
and (via scripts/generate_v1_v2_heatmaps.py):
    results/figures/v1_v2_scale_heatmap_{bleu,chrf2,comet}.{png,pdf,svg}

Example
-------
  python scripts/run_v1_v2_scale_ablation.py \\
      --fractions 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \\
      --repetitions 1 2 3 --seed 42

  # smoke
  python scripts/run_v1_v2_scale_ablation.py --fractions 0.0 0.5 1.0 --repetitions 1 2 \\
      --epochs 1 --max-train-samples 200 --max-val-samples 48 --max-test-samples 64 --no-comet
"""
import _paths  # noqa: F401

import argparse
import csv
import logging
import os
import sys

from src.config import RunConfig, load_experiment_yaml
from src.data.buckets import BUCKET_LABELS_LONG  # noqa: F401 (kept for parity)
from src.data.v1v2 import compose
from src.experiments.train_run import run_training
from src.utils.io import ensure_dir, read_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("v1_v2_scale")

CSV_FIELDS = ["repetitions", "v2_fraction", "training_examples", "bleu", "chrf2",
              "comet", "length_error", "seed", "status", "run_dir"]


def _fmt_alpha(a: float) -> str:
    return f"{a:.1f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", default="v1_v2_scale")
    ap.add_argument("--dataset", default=None,
                    help="dataset version supplying the shared val/test set (default from YAML: v2)")
    ap.add_argument("--v1-dir", default=None, help="V1 corpus dir (default from YAML: data/v1)")
    ap.add_argument("--v2-dir", default=None, help="V2 corpus dir (default from YAML: data/v2)")
    ap.add_argument("--config", default=None, help="downstream X-ABCD (default from YAML: X-0000)")
    ap.add_argument("--fractions", nargs="*", type=float, default=None)
    ap.add_argument("--repetitions", nargs="*", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--experiments-root", default="experiments")
    ap.add_argument("--out-csv", default="results/csv/v1_v2_scale_ablation.csv")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-completed", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-heatmaps", action="store_true")
    # smoke knobs
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--max-train-samples", type=int)
    ap.add_argument("--max-val-samples", type=int)
    ap.add_argument("--max-test-samples", type=int)
    ap.add_argument("--no-comet", action="store_true")
    args = ap.parse_args()

    exp = load_experiment_yaml(args.experiment)
    grid = exp.get("grid", {})
    fractions = args.fractions or grid.get("v2_fractions", [round(i / 10, 1) for i in range(11)])
    repetitions = args.repetitions or grid.get("repetitions", [1, 2, 3])
    seed = args.seed if args.seed is not None else exp.get("seed", 42)
    dataset = args.dataset or exp.get("dataset_version", "v2")   # val/test come from here
    downstream = (args.config or exp.get("downstream_config", "X-0000")).upper()
    run_group = exp.get("run_group", "v1_v2_scale")
    model_key = exp.get("model_key", "nllb")
    v1_dir = args.v1_dir or exp.get("v1_dir", "data/v1")
    v2_dir = args.v2_dir or exp.get("v2_dir", "data/v2")
    _ev = exp.get("evaluation") or {}
    metrics = _ev.get("metrics", ["bleu", "chrf2", "comet"])
    eval_corpora = _ev.get("eval_corpora", ["v2"])
    eval_variants = _ev.get("test_variants", ["unsampled"])
    run_comet = not args.no_comet

    overrides = dict(exp.get("defaults") or {})
    if args.epochs is not None:
        overrides["num_train_epochs"] = args.epochs
    for k, a in (("max_train_samples", args.max_train_samples),
                 ("max_val_samples", args.max_val_samples),
                 ("max_test_samples", args.max_test_samples)):
        if a is not None:
            overrides[k] = a

    ensure_dir(os.path.dirname(os.path.abspath(args.out_csv)))
    print(f"\n[v1_v2_scale] downstream={downstream} seed={seed}  eval set = dataset '{dataset}'")
    print(f"[v1_v2_scale] repetitions={repetitions}  v2_fractions={fractions}")
    print(f"[v1_v2_scale] V1 = {v1_dir}/train.*  (fixed)   V2 = {v2_dir}/train.*  (nested subsets)")

    # preview the deterministic composition sizes
    for rep in repetitions:
        for a in fractions:
            _, _, meta = compose(a, rep, v1_dir, v2_dir, subset_seed=seed)
            print(f"  rep={rep} alpha={_fmt_alpha(a)}  V1={meta.n_v1}  +V2={meta.n_v2_used}"
                  f"/{meta.n_v2_pool}  composed={meta.n_composed}  train_examples={meta.training_examples}")
    if args.dry_run:
        print("dry run — nothing executed.")
        return

    rows = []
    n_cells = len(repetitions) * len(fractions)
    i = 0
    for rep in repetitions:
        for a in fractions:
            i += 1
            comp_src, comp_tgt, meta = compose(a, rep, v1_dir, v2_dir, subset_seed=seed)
            cfg = RunConfig.resolve(
                downstream, seed, dataset, model_key, args.experiment,
                run_group=os.path.join(run_group, f"rep{rep}", f"alpha{_fmt_alpha(a)}"),
                experiments_root=args.experiments_root,
                overrides={**overrides, "train_repeat_factor": rep},
            )
            log.info("[%d/%d] rep=%d alpha=%s -> %s", i, n_cells, rep, _fmt_alpha(a), cfg.output_dir)
            status = "done"
            m = {}
            try:
                res = run_training(
                    cfg, skip_completed=args.skip_completed, run_final_eval=True,
                    eval_metrics=metrics, eval_corpora=eval_corpora, eval_variants=eval_variants,
                    run_comet=run_comet, device=args.device,
                    train_pairs_override=(comp_src, comp_tgt), verify_dataset=False,
                    manifest_extra={"analysis_only": True,
                                    "test_set_used_for_selection": False,
                                    "v1_v2_composition": meta.to_dict()},
                )
                if res["status"] not in ("completed", "skipped"):
                    status = res["status"]
                # CSV tracks the headline number: v2 / sampled (same as main experiments)
                ev = os.path.join(cfg.output_dir, "evaluation", "v2", "unsampled", "metrics.json")
                if os.path.isfile(ev):
                    m = read_json(ev)["metrics"]
            except Exception as exc:
                log.exception("cell rep=%d alpha=%s failed", rep, a)
                status = f"error: {exc}"

            rows.append({
                "repetitions": rep,
                "v2_fraction": _fmt_alpha(a),
                "training_examples": meta.training_examples,
                "bleu": m.get("bleu"),
                "chrf2": m.get("chrf2"),
                "comet": m.get("comet"),
                "length_error": m.get("absolute_length_error"),
                "seed": seed,
                "status": status,
                "run_dir": cfg.output_dir,
            })
            with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                w.writeheader()
                w.writerows(rows)
            log.info("  -> BLEU=%s chrF2=%s COMET=%s  [%s]", m.get("bleu"), m.get("chrf2"),
                     m.get("comet"), status)

    print(f"\n[v1_v2_scale] results -> {args.out_csv}")
    if not args.no_heatmaps:
        from subprocess import run as _run

        _run([sys.executable, os.path.join(os.path.dirname(__file__), "generate_v1_v2_heatmaps.py"),
              "--csv", args.out_csv], check=False)


if __name__ == "__main__":
    main()
