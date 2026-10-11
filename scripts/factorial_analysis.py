#!/usr/bin/env python
"""
Fit the replicated 2^4 factorial model (brief section 18).

Reads per-run test metrics from the experiment tree, fits the saturated model
(A=BBS B=TLC C=RoPE D=ALiBi, +/-1 coding) by OLS over the 3-seed replicates,
and writes, per metric:

    results/csv/factorial_effects_<metric>.csv
    results/figures/factorial_effects_<metric>.{png,pdf}

Interaction wording stays cautious ("positive/negative interaction",
"significant"); never "super-additive" (brief sections 18/36).

Example:
  python scripts/factorial_analysis.py --dataset v2 --metrics bleu comet chrf2
"""
import _paths  # noqa: F401

import argparse
import csv
import os
import sys

from src.statistics.aggregate import collect_runs
from src.statistics.factorial import fit
from src.visualization.plots import factorial_effects_plot


def _metric_by_config(records, metric, seeds=None):
    out = {}
    for r in records:
        if seeds and r["seed"] not in seeds:
            continue
        if r["status"] not in (None, "completed"):
            continue
        v = r["metrics"].get(metric)
        out.setdefault(r["case_id"], []).append(v)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="v2")
    ap.add_argument("--experiments-root", default="experiments")
    ap.add_argument("--run-group", default="factorial")
    ap.add_argument("--test-corpus", default="v2", choices=["v2", "v1"])
    ap.add_argument("--test-variant", default="unsampled", choices=["sampled", "unsampled"])
    ap.add_argument("--metrics", nargs="*", default=["bleu", "chrf2", "comet"])
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--csv-dir", default="results/csv")
    ap.add_argument("--fig-dir", default="results/figures")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    records = collect_runs(args.experiments_root, args.dataset, args.run_group,
                           args.test_corpus, args.test_variant)
    if not records:
        sys.exit(f"no evaluated runs under {args.experiments_root}/{args.dataset}/{args.run_group}")

    os.makedirs(args.csv_dir, exist_ok=True)
    for metric in args.metrics:
        mbc = _metric_by_config(records, metric, seeds=args.seeds)
        try:
            effects = fit(mbc, alpha=args.alpha)
        except ValueError as exc:
            print(f"[factorial] {metric}: SKIPPED — {exc}")
            continue
        csv_path = os.path.join(args.csv_dir, f"factorial_effects_{metric}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(effects[0].to_dict().keys()))
            w.writeheader()
            for e in effects:
                w.writerow(e.to_dict())
        # canonical name for the primary metric too
        if metric == args.metrics[0]:
            with open(os.path.join(args.csv_dir, "factorial_effects.csv"), "w", newline="",
                      encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(effects[0].to_dict().keys()))
                w.writeheader()
                for e in effects:
                    w.writerow(e.to_dict())
        fig_paths = factorial_effects_plot(csv_path,
                                           os.path.join(args.fig_dir, f"factorial_effects_{metric}"),
                                           metric_label=metric.upper())
        if metric == args.metrics[0]:
            factorial_effects_plot(csv_path, os.path.join(args.fig_dir, "factorial_effects"),
                                   metric_label=metric.upper())
        sig = [e.term for e in effects if e.significant]
        print(f"[factorial] {metric}: CSV -> {csv_path}")
        print(f"            figures -> {fig_paths}")
        print(f"            significant terms (p<{args.alpha}): {sig or 'none'}")


if __name__ == "__main__":
    main()
