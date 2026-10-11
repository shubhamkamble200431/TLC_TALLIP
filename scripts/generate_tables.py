#!/usr/bin/env python
"""
Generate every LaTeX table from the result CSVs (brief section 24).

    table_3seed_overall.tex    table_3seed_bucket.tex     table_bbs_ablation.tex
    table_mbart_baseline.tex   table_factorial_effects.tex table_domain_length.tex
    table_length_metrics.tex

Nothing is hand-typed; missing inputs are skipped with a note.

Example:
  python scripts/generate_tables.py --csv-dir results/csv --out-dir results/tables
"""
import _paths  # noqa: F401

import argparse
import os

from src.reporting import tables as T


def _exists(*paths):
    return all(os.path.isfile(p) for p in paths)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-dir", default="results/csv")
    ap.add_argument("--out-dir", default="results/tables")
    ap.add_argument("--bucket-metric", default="bleu")
    ap.add_argument("--factorial-metric", default="bleu")
    args = ap.parse_args()

    c = args.csv_dir
    summary = os.path.join(c, "results_3seed_summary.csv")
    buckets = os.path.join(c, "bucket_results.csv")
    made = []

    if _exists(summary):
        made.append(T.table_3seed_overall(summary, args.out_dir))
        made.append(T.table_length_metrics(summary, args.out_dir))
    if _exists(buckets):
        made.append(T.table_3seed_bucket(buckets, args.out_dir, metric=args.bucket_metric))

    bbs = os.path.join(c, "bbs_weight_ablation.csv")
    if _exists(bbs):
        made.append(T.table_bbs_ablation(bbs, args.out_dir))

    mbart = os.path.join(c, "mbart50", "results_3seed_summary.csv")
    if not os.path.isfile(mbart):
        mbart = os.path.join(c, "mbart50_summary.csv")
    if _exists(summary) and os.path.isfile(mbart):
        made.append(T.table_mbart_baseline(summary, mbart, args.out_dir))

    fx = os.path.join(c, f"factorial_effects_{args.factorial_metric}.csv")
    if not os.path.isfile(fx):
        fx = os.path.join(c, "factorial_effects.csv")
    if os.path.isfile(fx):
        made.append(T.table_factorial_effects(fx, args.out_dir, metric_label=args.factorial_metric.upper()))

    dom = os.path.join(c, "domain_length.csv")
    if _exists(dom):
        made.append(T.table_domain_length(dom, args.out_dir, metric=args.bucket_metric))

    print("[tables] generated:")
    for p in made:
        print(f"  {p}")
    if not made:
        print("  (nothing — run evaluate_all.py / factorial_analysis.py first)")


if __name__ == "__main__":
    main()
