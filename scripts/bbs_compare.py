#!/usr/bin/env python
"""
Manual vs grid-selected BBS comparison + the BBS weight-ablation table
(brief sections 10, 11).

For each downstream config (default X-1100, X-1101) and each seed, trains with
  * BBS-manual        = [1.0, 1.0, 1.2, 1.5, 2.0]
  * BBS-grid-selected = the vector frozen by run_bbs_search.py
(and optionally a few alternative vectors), everything else identical, then
evaluates once on the held-out TEST set and writes:

    experiments/<ds>/bbs_search/compare/<cfg>/<variant>/seed_<n>/...
    results/csv/bbs_weight_ablation.csv    (uniform / manual / grid-selected /
                                            alternatives / best-val / selected)

Example:
  python scripts/bbs_compare.py --dataset v2 --configs X-1100 X-1101 --seeds 42 43 44
"""
import _paths  # noqa: F401

import argparse
import csv
import os
import sys

from src.config import RunConfig, load_experiment_yaml
from src.data.buckets import BUCKET_LABELS_LONG
from src.data.dataset import normalized_bucket_weights
from src.experiments.train_run import run_training
from src.utils.io import read_json

MANUAL = [1.0, 1.0, 1.2, 1.5, 2.0]
UNIFORM = [1.0, 1.0, 1.0, 1.0, 1.0]


def _mean_test_metrics(exp_root, ds, group, cfg_id, variant, seeds):
    b5 = BUCKET_LABELS_LONG[5]
    acc = {}
    for s in seeds:
        mp = os.path.join(exp_root, ds, group, "compare", cfg_id, variant, f"seed_{s}",
                          "evaluation", "v2", "unsampled", "metrics.json")
        if not os.path.isfile(mp):
            continue
        m = read_json(mp)["metrics"]
        for k, v in {
            "bleu": m.get("bleu"), "chrf2": m.get("chrf2"), "comet": m.get("comet"),
            "b5_bleu": m.get(f"{b5}_bleu"),
            "length_error": m.get("absolute_length_error"),
        }.items():
            acc.setdefault(k, []).append(v)
    import numpy as np

    out = {}
    for k, vals in acc.items():
        vals = [x for x in vals if x is not None]
        out[f"{k}_mean"] = float(np.mean(vals)) if vals else None
        out[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="v2")
    ap.add_argument("--experiment", default="bbs_grid")
    ap.add_argument("--experiments-root", default="experiments")
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-completed", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--max-train-samples", type=int)
    ap.add_argument("--max-val-samples", type=int)
    ap.add_argument("--max-test-samples", type=int)
    ap.add_argument("--out", default="results/csv/bbs_weight_ablation.csv")
    args = ap.parse_args()

    exp = load_experiment_yaml(args.experiment)
    cmp_cfg = exp.get("compare", {})
    configs = args.configs or cmp_cfg.get("configs", ["X-1100", "X-1101"])
    seeds = args.seeds or cmp_cfg.get("seeds", [42, 43, 44])
    group = exp.get("run_group", "bbs_search")
    model_key = exp.get("model_key", "nllb")
    budget_exp = cmp_cfg.get("full_budget_from", "factorial")
    base_defaults = dict(load_experiment_yaml(budget_exp).get("defaults", {}))

    sel_path = os.path.join(args.experiments_root, args.dataset, "bbs_search", "selected_bbs_weights.json")
    selected = read_json(sel_path)["selected_vector_normalized"] if os.path.isfile(sel_path) else None
    if selected is None:
        print(f"[bbs_compare] WARNING: {sel_path} not found — run run_bbs_search.py first. "
              f"Using manual as a placeholder for 'grid_selected'.")
        selected = MANUAL

    variants = {
        "uniform": normalized_bucket_weights(UNIFORM),
        "manual": normalized_bucket_weights(MANUAL),
        "grid_selected": normalized_bucket_weights(selected),
    }
    for name, vec in (exp.get("compare", {}).get("alternatives", {}) or {}).items():
        variants[name] = normalized_bucket_weights(vec)

    ov_common = dict(base_defaults)
    ov_common["use_bucket_balanced_sampling"] = True
    if args.epochs is not None:
        ov_common["num_train_epochs"] = args.epochs
    for k, a in (("max_train_samples", args.max_train_samples),
                 ("max_val_samples", args.max_val_samples),
                 ("max_test_samples", args.max_test_samples)):
        if a is not None:
            ov_common[k] = a

    print(f"[bbs_compare] configs={configs} seeds={seeds} variants={list(variants)}")
    if args.dry_run:
        for cid in configs:
            for var, vec in variants.items():
                print(f"  {cid} {var:<14} -> {[round(x,3) for x in vec]}")
        return

    for cid in configs:
        for var, vec in variants.items():
            for s in seeds:
                cfg = RunConfig.resolve(
                    cid, s, args.dataset, model_key, args.experiment,
                    run_group=os.path.join(group, "compare", cid, var),
                    experiments_root=args.experiments_root,
                    overrides={**ov_common, "bucket_weights": vec},
                )
                res = run_training(cfg, skip_completed=args.skip_completed, run_final_eval=True,
                                   eval_metrics=["bleu", "chrf2", "ter", "meteor", "bertscore", "comet"],
                                   device=args.device)
                print(f"  {cid} {var} seed={s}: {res['status']}")

    # ---- assemble the ablation table CSV -------------------------
    rows = []
    for cid in configs:
        for var, vec in variants.items():
            agg = _mean_test_metrics(args.experiments_root, args.dataset, group, cid, var, seeds)
            rows.append({
                "config": cid, "variant": var, "eval_split": "test",
                **{f"w{i+1}": vec[i] for i in range(5)},
                **agg,
            })

    # also fold in a few strong ALTERNATIVE candidates from the search itself
    # (brief section 11: "Do not report only the winning candidate.") — these
    # carry VALIDATION metrics and are labelled as such.
    search_csv = os.path.join(args.experiments_root, args.dataset, "bbs_search", "bbs_search_results.csv")
    if os.path.isfile(search_csv):
        import csv as _csv

        with open(search_csv) as f:
            srows = [r for r in _csv.DictReader(f) if r.get("val_comet") not in (None, "", "None")]
        srows.sort(key=lambda r: float(r["val_comet"]), reverse=True)
        for rank, r in enumerate(srows[:4]):
            rows.append({
                "config": "(search)", "variant": f"search_top{rank+1}_{r.get('tag','')}",
                "eval_split": "validation",
                **{f"w{i+1}": float(r[f"normalized_w{i+1}"]) for i in range(5)},
                "bleu_mean": float(r["val_bleu"]) if r.get("val_bleu") not in ("", "None") else None,
                "chrf2_mean": float(r["val_chrf2"]) if r.get("val_chrf2") not in ("", "None") else None,
                "comet_mean": float(r["val_comet"]),
                "length_error_mean": float(r["val_length_error"]) if r.get("val_length_error") not in ("", "None") else None,
            })
    if rows:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        allkeys = set()
        for r in rows:
            allkeys.update(r)
        keys = ["config", "variant", "eval_split", "w1", "w2", "w3", "w4", "w5"] + \
               sorted(k for k in allkeys if k not in ("config", "variant", "eval_split")
                      and not (len(k) == 2 and k[0] == "w"))
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"[bbs_compare] ablation table -> {args.out}")


if __name__ == "__main__":
    main()
