#!/usr/bin/env python
"""
Regenerate the V1 + alpha*V2 data-scale heatmaps from the saved CSV.

    rows    = number of repetitions
    columns = V1+0.0V2, V1+0.1V2, ..., V1+1.0V2
    cell    = test-set score (exact number annotated in every cell)

Produces (>=300 DPI PNG + PDF + SVG), driven entirely by
results/csv/v1_v2_scale_ablation.csv — no hard-coded results:

    results/figures/v1_v2_scale_heatmap_bleu.{png,pdf,svg}
    results/figures/v1_v2_scale_heatmap_chrf2.{png,pdf,svg}
    results/figures/v1_v2_scale_heatmap_comet.{png,pdf,svg}

Example:
  python scripts/generate_v1_v2_heatmaps.py
"""
import _paths  # noqa: F401

import argparse
import os

import numpy as np
import pandas as pd

from src.config import load_experiment_yaml
from src.visualization.heatmaps import annotated_heatmap

_METRIC_TITLE = {"bleu": "BLEU", "chrf2": "chrF2", "comet": "COMET",
                 "length_error": "abs. length error"}


def _pivot(df: pd.DataFrame, metric: str):
    reps = sorted(df["repetitions"].unique())
    fracs = sorted(df["v2_fraction"].unique(), key=float)
    mat = np.full((len(reps), len(fracs)), np.nan)
    for r_i, r in enumerate(reps):
        for c_i, a in enumerate(fracs):
            sub = df[(df["repetitions"] == r) & (df["v2_fraction"] == a)]
            if not sub.empty and metric in sub.columns:
                v = pd.to_numeric(sub[metric].iloc[0], errors="coerce")
                if pd.notna(v):
                    mat[r_i, c_i] = float(v)
    row_labels = [f"{r} rep" if r == 1 else f"{r} reps" for r in reps]
    col_labels = [f"V1+{float(a):.1f}V2" for a in fracs]
    return mat, row_labels, col_labels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="results/csv/v1_v2_scale_ablation.csv")
    ap.add_argument("--fig-dir", default="results/figures")
    ap.add_argument("--metrics", nargs="*", default=["bleu", "chrf2", "comet"])
    ap.add_argument("--cmap", default=None)
    ap.add_argument("--dpi", type=int, default=None)
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        raise SystemExit(f"{args.csv} not found — run scripts/run_v1_v2_scale_ablation.py first.")

    hm_cfg = (load_experiment_yaml("v1_v2_scale").get("heatmap") or {})
    cmap = args.cmap or hm_cfg.get("colormap", "viridis")
    dpi = args.dpi or int(hm_cfg.get("dpi", 300))
    formats = tuple(hm_cfg.get("formats", ["png", "pdf", "svg"]))

    df = pd.read_csv(args.csv, dtype={"v2_fraction": str})
    made = []
    for metric in args.metrics:
        if metric not in df.columns:
            print(f"[skip] {metric} not in {args.csv}")
            continue
        mat, rows, cols = _pivot(df, metric)
        if not np.isfinite(mat).any():
            print(f"[skip] {metric}: no finite values yet")
            continue
        lower_better = metric == "length_error"
        made += annotated_heatmap(
            mat, rows, cols,
            os.path.join(args.fig_dir, f"v1_v2_scale_heatmap_{metric}"),
            title=f"V1 + $\\alpha\\cdot$V2 data-scale ablation — test {_METRIC_TITLE.get(metric, metric)}",
            xlabel="training-set composition  (V1 fixed; $\\alpha$ = fraction of V2 added)",
            ylabel="training-data repetitions",
            cbar_label=f"test {_METRIC_TITLE.get(metric, metric)}"
                       + (" (lower is better)" if lower_better else ""),
            cmap=cmap, value_fmt="{:.2f}", dpi=dpi, formats=formats,
            figsize=(1.05 * mat.shape[1] + 2.2, 0.6 * mat.shape[0] + 2.0),
        )

    print("[v1_v2 heatmaps] generated:")
    for p in made:
        print(f"  {p}")
    if not made:
        print("  (nothing — CSV has no finished cells yet)")


if __name__ == "__main__":
    main()
