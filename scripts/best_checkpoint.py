#!/usr/bin/env python
"""
Pick the single best trained model and print its checkpoint path.

Reads results/csv/results_3seed_summary.csv (produced by the stats phase), ranks
configurations by a chosen metric's 3-seed mean, and prints the path to that
configuration's checkpoint for a given seed:

    experiments/<dataset>/<run_group>/<CONFIG>/seed_<seed>/checkpoint-best

Designed for shell capture:

    CKPT=$(python scripts/best_checkpoint.py --dataset v2 --metric bleu)
    python scripts/run_bucket_repeat_ablation.py --checkpoint "$CKPT" \
        --config "$(python scripts/best_checkpoint.py --dataset v2 --print config)" --dataset v2

`--print` selects what goes to stdout: path (default) | config | seed | row | ranking.
The human-readable ranking always goes to stderr unless --quiet.
"""
import _paths  # noqa: F401

import argparse
import csv
import os
import sys

LOWER_BETTER = {"ter", "absolute_length_error", "relative_length_error",
                "premature_eos_rate", "overgeneration_rate", "truncation_rate"}


def _load(summary_csv):
    if not os.path.isfile(summary_csv):
        raise SystemExit(f"{summary_csv} not found — run the stats phase "
                         f"(scripts/aggregate_results.py / 70_stats.slurm) first.")
    with open(summary_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{summary_csv} is empty.")
    return rows


def _rank(rows, metric):
    col = f"{metric}_mean"
    if col not in rows[0]:
        raise SystemExit(f"column {col!r} not in summary CSV; "
                         f"available: {sorted(k for k in rows[0] if k.endswith('_mean'))}")

    def key(r):
        try:
            v = float(r[col])
        except (TypeError, ValueError):
            v = float("nan")
        return v

    ranked = [r for r in rows if key(r) == key(r)]  # drop NaN
    ranked.sort(key=key, reverse=metric not in LOWER_BETTER)
    return ranked, col


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary-csv", default="results/csv/results_3seed_summary.csv")
    ap.add_argument("--dataset", default="v2")
    ap.add_argument("--run-group", default="factorial")
    ap.add_argument("--experiments-root", default="experiments")
    ap.add_argument("--metric", default="bleu", help="rank configs by this metric's 3-seed mean")
    ap.add_argument("--seed", type=int, default=42, help="which seed's checkpoint to point at")
    ap.add_argument("--checkpoint-name", default="checkpoint-best")
    ap.add_argument("--print", dest="what", default="path",
                    choices=["path", "config", "seed", "row", "ranking"])
    ap.add_argument("--require-exists", action="store_true",
                    help="fail if the resolved checkpoint directory is missing")
    ap.add_argument("--quiet", action="store_true", help="no ranking on stderr")
    args = ap.parse_args()

    rows = _load(args.summary_csv)
    ranked, col = _rank(rows, args.metric)
    if not ranked:
        raise SystemExit(f"no usable {col} values in {args.summary_csv}")
    best = ranked[0]
    cfg = best["config"]
    ckpt = os.path.join(args.experiments_root, args.dataset, args.run_group,
                        cfg, f"seed_{args.seed}", args.checkpoint_name)

    if not args.quiet:
        print(f"# ranking by {args.metric} ({col}), {args.dataset}/{args.run_group}:", file=sys.stderr)
        for i, r in enumerate(ranked, 1):
            mark = "  <-- best" if i == 1 else ""
            std = r.get(f"{args.metric}_std", "")
            print(f"#  {i:>2}. {r['config']:<8} {float(r[col]):8.3f}  "
                  f"(sd {float(std):.3f}){mark}" if std not in ("", None)
                  else f"#  {i:>2}. {r['config']:<8} {float(r[col]):8.3f}{mark}", file=sys.stderr)
        print(f"# checkpoint: {ckpt}", file=sys.stderr)

    if args.require_exists and not os.path.isdir(ckpt):
        raise SystemExit(f"resolved checkpoint does not exist: {ckpt}")

    if args.what == "path":
        print(ckpt)
    elif args.what == "config":
        print(cfg)
    elif args.what == "seed":
        print(args.seed)
    elif args.what == "row":
        print({k: best[k] for k in ("config", "seeds", col)})
    elif args.what == "ranking":
        for i, r in enumerate(ranked, 1):
            print(f"{i}\t{r['config']}\t{float(r[col]):.4f}")


if __name__ == "__main__":
    main()
