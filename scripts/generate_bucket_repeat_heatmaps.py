#!/usr/bin/env python
"""
Heatmaps for the bucket-cap x repetition inference ablation.

    rows    = repetitions (1..10)
    columns = k sentences per bucket, labelled  "<f>M=<k>"  (0.2M .. 1.0M)
    cell    = metric over the 5-bucket-equal test set

One heatmap per metric (micro/corpus and macro/mean-of-buckets), per dataset.
Reproducible from results/csv/bucket_repeat_ablation_<ds>.csv alone.

Usage:
  python scripts/generate_bucket_repeat_heatmaps.py --dataset v2
"""
import _paths  # noqa: F401

import argparse
import os

import numpy as np
import pandas as pd

from src.config import load_experiment_yaml
from src.visualization.heatmaps import annotated_heatmap

_TITLE = {"bleu": "BLEU", "chrf2": "chrF2", "comet": "COMET",
          "bleu_macro": "macro BLEU", "chrf2_macro": "macro chrF2", "comet_macro": "macro COMET"}


def _heat(df, metric, dataset, fig_dir, cmap, dpi, formats):
    if metric not in df.columns or df[metric].dropna().empty:
        print(f"[skip] {metric}: no data")
        return []
    reps = sorted(df["repetitions"].unique())
    cols = sorted(df["k_per_bucket"].unique())
    frac_of = dict(zip(df["k_per_bucket"], df["fraction"]))
    mat = np.full((len(reps), len(cols)), np.nan)
    for i, r in enumerate(reps):
        for j, k in enumerate(cols):
            sub = df[(df["repetitions"] == r) & (df["k_per_bucket"] == k)]
            if not sub.empty:
                mat[i, j] = pd.to_numeric(sub[metric].iloc[0], errors="coerce")
    col_labels = [f"{frac_of[k]:.1f}M\n={int(k)}" for k in cols]
    row_labels = [f"{r} rep" if r == 1 else f"{r} reps" for r in reps]
    lower_better = metric.startswith("length")
    return annotated_heatmap(
        mat, row_labels, col_labels,
        os.path.join(fig_dir, f"bucket_repeat_heatmap_{dataset}_{metric}"),
        title=f"{dataset}: bucket-equal test — {_TITLE.get(metric, metric)}  "
              f"(M = min bucket count = {int(df['M'].iloc[0])})",
        xlabel="sentences per bucket  (fraction of M  =  k)",
        ylabel="repetitions per bucket",
        cbar_label=_TITLE.get(metric, metric) + (" (lower better)" if lower_better else ""),
        cmap=cmap, value_fmt="{:.2f}", dpi=dpi, formats=formats,
        figsize=(1.05 * len(cols) + 2.4, 0.5 * len(reps) + 1.8),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--dataset", default="v2", choices=["v2", "v1"])
    ap.add_argument("--fig-dir", default="results/figures")
    ap.add_argument("--metrics", nargs="*", default=["bleu", "chrf2", "comet"])
    ap.add_argument("--cmap", default=None)
    ap.add_argument("--dpi", type=int, default=None)
    args = ap.parse_args()

    csv = args.csv or f"results/csv/bucket_repeat_ablation_{args.dataset}.csv"
    if not os.path.isfile(csv):
        raise SystemExit(f"{csv} not found — run scripts/run_bucket_repeat_ablation.py first.")

    hm = load_experiment_yaml("bucket_repeat").get("heatmap", {})
    cmap = args.cmap or hm.get("colormap", "viridis")
    dpi = args.dpi or int(hm.get("dpi", 300))
    formats = tuple(hm.get("formats", ["png", "pdf"]))

    df = pd.read_csv(csv)
    made = []
    for m in args.metrics:
        made += _heat(df, m, args.dataset, args.fig_dir, cmap, dpi, formats)
        made += _heat(df, f"{m}_macro", args.dataset, args.fig_dir, cmap, dpi, formats)

    print("[bucket_repeat heatmaps] generated:")
    for p in made:
        print(f"  {p}")


if __name__ == "__main__":
    main()
