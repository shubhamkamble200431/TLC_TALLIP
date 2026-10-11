#!/usr/bin/env python
"""
BBS weight-selection search (brief sections 6-11, 37-P3/P4).

VALIDATION-ONLY grid search over *relative* bucket emphasis.  Every candidate
vector is mean-normalised (sum = 5).  The winner is chosen purely on the
validation set; only then is it frozen for the held-out test evaluation
(done later by run_factorial / the compare step).

Outputs (under experiments/<dataset>/bbs_search/):
    bbs_search_results.csv      -- one row per candidate  (section 8 schema)
    selected_bbs_weights.json   -- frozen winner + provenance

Examples
--------
  python scripts/run_bbs_search.py --dataset v2 --config X-1000 \\
      --selection-metric comet

  # tiny smoke search
  python scripts/run_bbs_search.py --config X-1000 --epochs 1 \\
      --max-train-samples 200 --max-val-samples 64 --max-candidates 3
"""
import _paths  # noqa: F401

import argparse
import csv
import logging
import os
import sys
from datetime import datetime, timezone

from src.config import RunConfig, load_experiment_yaml
from src.evaluation.evaluate import evaluate_checkpoint
from src.experiments.bbs_grid import from_config as space_from_config
from src.experiments.train_run import run_training
from src.utils.io import ensure_dir, write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("run_bbs_search")

_PRIMARY = {  # selection-metric -> (val_key, lower_is_better)
    "comet": ("val_comet", False),
    "bleu": ("val_bleu", False),
    "chrf2": ("val_chrf2", False),
    "length_error": ("val_length_error", True),
}


def _val_scores(cfg: RunConfig, best_dir: str, device: str, run_comet: bool) -> dict:
    payload = evaluate_checkpoint(
        cfg, checkpoint_path=best_dir, split="val",
        num_beams=cfg.final_eval_num_beams, run_comet=run_comet, device=device,
        max_samples=cfg.max_val_samples,
        metrics=["bleu", "chrf2", "comet"],
    )
    m = payload["metrics"]
    return {
        "val_bleu": m.get("bleu"),
        "val_chrf2": m.get("chrf2"),
        "val_comet": m.get("comet"),
        "val_length_error": m.get("absolute_length_error"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="v2")
    ap.add_argument("--experiment", default="bbs_grid")
    ap.add_argument("--config", default=None, help="downstream X-ABCD config (default from YAML)")
    ap.add_argument("--selection-metric", default=None, choices=list(_PRIMARY),
                    help="validation metric to select on (default from YAML)")
    ap.add_argument("--experiments-root", default="experiments")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-completed", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--max-candidates", type=int)
    ap.add_argument("--max-train-samples", type=int)
    ap.add_argument("--max-val-samples", type=int)
    ap.add_argument("--no-comet", action="store_true")
    ap.add_argument("--full-train", action="store_true",
                    help="ignore any max-train-samples; search on the entire train set")
    ap.add_argument("--no-stratified", action="store_true",
                    help="use a plain first-N slice instead of a bucket-stratified subsample")
    # --- parallelisation across GPUs (run several instances) ---
    ap.add_argument("--out-csv", default=None, help="override the results CSV path (for parallel slices)")
    ap.add_argument("--candidate-mod", default=None,
                    help='"N/R" -> only run candidates where index %% N == R')
    ap.add_argument("--no-select", action="store_true",
                    help="skip writing selected_bbs_weights.json (parallel slice)")
    ap.add_argument("--select-from", default=None,
                    help="glob of per-slice CSVs -> merge, rank, write selected_bbs_weights.json; no training")
    args = ap.parse_args()

    exp = load_experiment_yaml(args.experiment)
    grid_cfg = dict(exp.get("grid", {}))
    if args.max_candidates is not None:
        grid_cfg["max_candidates"] = args.max_candidates
    space = space_from_config(grid_cfg)
    candidates = space.generate()

    downstream = (args.config or exp.get("downstream_config", "X-1000")).upper()
    sel_name = args.selection_metric or exp.get("selection", {}).get("primary_metric", "val_comet")
    sel_name = sel_name.replace("val_", "")
    sel_key, lower_better = _PRIMARY[sel_name]
    search_seeds = exp.get("bbs_search_seeds", [42])
    run_group = exp.get("run_group", "bbs_search")

    overrides = {"use_bucket_balanced_sampling": True}
    for k, v in (exp.get("defaults") or {}).items():
        overrides[k] = v
    if args.epochs is not None:
        overrides["num_train_epochs"] = args.epochs
    if args.max_train_samples is not None:
        overrides["max_train_samples"] = args.max_train_samples
    if args.max_val_samples is not None:
        overrides["max_val_samples"] = args.max_val_samples
    if args.full_train:
        overrides["max_train_samples"] = None
    if args.no_stratified:
        overrides["stratified_train_subsample"] = False
    run_comet = not args.no_comet

    out_root = ensure_dir(os.path.join(args.experiments_root, args.dataset, run_group))
    results_csv = args.out_csv or os.path.join(out_root, "bbs_search_results.csv")
    selected_json = os.path.join(out_root, "selected_bbs_weights.json")

    fieldnames = [
        "candidate_id", "tag",
        "w1", "w2", "w3", "w4", "w5",
        "normalized_w1", "normalized_w2", "normalized_w3", "normalized_w4", "normalized_w5",
        "val_bleu", "val_chrf2", "val_comet", "val_length_error",
        "training_steps", "seed", "status",
    ]

    # ---- merge-and-select mode (no training) ----------------------
    if args.select_from:
        import glob as _glob

        merged = []
        for path in sorted(_glob.glob(args.select_from)):
            with open(path, encoding="utf-8") as f:
                merged.extend(list(csv.DictReader(f)))
        for r in merged:
            for k in ("val_bleu", "val_chrf2", "val_comet", "val_length_error"):
                r[k] = float(r[k]) if r.get(k) not in (None, "", "None") else None
        with open(results_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(merged)
        _select_and_write(merged, sel_key, lower_better, selected_json, results_csv,
                          space, downstream, args, run_group, len(candidates))
        return

    if args.candidate_mod:
        n, r = (int(x) for x in args.candidate_mod.split("/"))
        candidates = [c for i, c in enumerate(candidates) if i % n == r]
        print(f"[bbs_search] candidate-mod {args.candidate_mod} -> {len(candidates)} candidates: "
              f"{[c.id for c in candidates]}")

    print(f"\n[bbs_search] downstream={downstream}  select on {sel_key} "
          f"({'lower' if lower_better else 'higher'} better)")
    print(f"[bbs_search] {len(candidates)} candidates x seeds {search_seeds}  mode={space.mode}")
    for c in candidates:
        print(f"  {c.id:<10} tag={c.tag:<20} normalized={[round(x,3) for x in c.normalized]}")
    if args.dry_run:
        print("dry run — nothing executed.")
        return

    rows = []
    for c in candidates:
        for seed in search_seeds:
            cfg = RunConfig.resolve(
                downstream, seed, args.dataset, exp.get("model_key", "nllb"), args.experiment,
                run_group=os.path.join(run_group, "candidates", c.id),
                experiments_root=args.experiments_root,
                overrides={**overrides, "bucket_weights": c.normalized},
            )
            status = "done"
            steps = None
            try:
                res = run_training(cfg, skip_completed=args.skip_completed, run_final_eval=False,
                                   device=args.device)
                best_dir = os.path.join(cfg.output_dir, "checkpoint-best")
                vs = _val_scores(cfg, best_dir, args.device, run_comet)
                mpath = os.path.join(cfg.output_dir, "metrics.json")
                if os.path.isfile(mpath):
                    from src.utils.io import read_json

                    steps = read_json(mpath).get("global_step")
            except Exception as exc:
                log.exception("candidate %s seed %s failed", c.id, seed)
                status = f"error: {exc}"
                vs = {k: None for k in ("val_bleu", "val_chrf2", "val_comet", "val_length_error")}
            row = {
                "candidate_id": c.id, "tag": c.tag,
                **{f"w{i+1}": c.raw[i] for i in range(5)},
                **{f"normalized_w{i+1}": c.normalized[i] for i in range(5)},
                **vs, "training_steps": steps, "seed": seed, "status": status,
            }
            rows.append(row)
            with open(results_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(rows)
            log.info("[bbs_search] %s seed=%s  %s=%s", c.id, seed, sel_key, row.get(sel_key))

    if args.no_select or args.candidate_mod:
        print(f"[bbs_search] slice done -> {results_csv}  (merge with --select-from)")
        return

    _select_and_write(rows, sel_key, lower_better, selected_json, results_csv,
                      space, downstream, args, run_group, len(candidates))


def _select_and_write(rows, sel_key, lower_better, selected_json, results_csv,
                      space, downstream, args, run_group, n_candidates):
    seeds = sorted({int(r["seed"]) for r in rows})
    s0 = seeds[0] if seeds else 42
    valid = [r for r in rows if r.get(sel_key) is not None and int(r["seed"]) == s0]
    if not valid and sel_key != "val_bleu":
        fb = [r for r in rows if r.get("val_bleu") is not None and int(r["seed"]) == s0]
        if fb:
            log.warning("[bbs_search] no %s values; falling back to val_bleu for selection", sel_key)
            sel_key, lower_better, valid = "val_bleu", False, fb
    if not valid:
        raise SystemExit("no valid candidate results to select from")
    best = min(valid, key=lambda r: r[sel_key]) if lower_better else max(valid, key=lambda r: r[sel_key])

    write_json(selected_json, {
        "selected_vector_raw": [float(best[f"w{i+1}"]) for i in range(5)],
        "selected_vector_normalized": [float(best[f"normalized_w{i+1}"]) for i in range(5)],
        "selection_metric": sel_key,
        "selection_lower_is_better": lower_better,
        "validation_score": best[sel_key],
        "candidate_id": best["candidate_id"],
        "seed": int(best["seed"]),
        "downstream_config": downstream,
        "search_space": {
            "values": space.values, "mode": space.mode, "monotonic": space.monotonic,
            "normalize_mean": space.normalize_mean, "coordinate_axes": space.coordinate_axes,
            "n_candidates": n_candidates,
        },
        "normalization_rule": "mean(w)=1  <=>  sum(w)=5",
        "results_csv": results_csv,
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "experiment_id": f"{args.dataset}/{run_group}/{downstream}",
        "test_set_used": False,
    })
    print(f"\n[bbs_search] SELECTED {best['candidate_id']} ({best.get('tag','')})  "
          f"normalized={[round(float(best[f'normalized_w{i+1}']),3) for i in range(5)]}  "
          f"{sel_key}={best[sel_key]}")
    print(f"[bbs_search] results  -> {results_csv}")
    print(f"[bbs_search] selected -> {selected_json}")


if __name__ == "__main__":
    main()
