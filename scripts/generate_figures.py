#!/usr/bin/env python
"""
Generate every publication figure from the result CSVs (brief section 23).

  1  BBS weight heatmap (COMET + BLEU + length error)   -> results/figures/bbs_weight_heatmap_*
  2  Config x length-bucket BLEU heatmap                -> config_bucket_bleu_heatmap
  3  Config x length-bucket COMET heatmap               -> config_bucket_comet_heatmap
  4  Mean BLEU vs source length                         -> bleu_vs_length
  5  Mean COMET vs source length                        -> comet_vs_length
  6  Length error vs source length                      -> length_error_vs_length
  7  Factorial main effects                             -> factorial_effects
  8  Factorial interaction plots                        -> interaction_*
  9  BBS validation performance vs weight config        -> bbs_validation_curve
  10 Seed variation / error-bar plot                    -> seed_variation_*

Every figure is reproducible from CSV, >=300 DPI PNG + vector PDF, colormap
from --cmap.  Missing inputs are skipped with a note.
"""
import _paths  # noqa: F401

import argparse
import os

from src.visualization import heatmaps as H
from src.visualization import plots as P


def _has(p):
    return os.path.isfile(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-dir", default="results/csv")
    ap.add_argument("--fig-dir", default="results/figures")
    ap.add_argument("--experiments-root", default="experiments")
    ap.add_argument("--dataset", default="v2")
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    c, fig = args.csv_dir, args.fig_dir
    os.makedirs(fig, exist_ok=True)
    made = []

    bbs_csv = os.path.join(args.experiments_root, args.dataset, "bbs_search", "bbs_search_results.csv")
    if _has(bbs_csv):
        for metric, stem in (("val_comet", "bbs_weight_heatmap_comet"),
                             ("val_bleu", "bbs_weight_heatmap_bleu"),
                             ("val_length_error", "bbs_weight_heatmap_length_error")):
            try:
                made += H.bbs_weight_heatmap(bbs_csv, os.path.join(fig, stem), metric=metric,
                                             cmap=args.cmap, dpi=args.dpi)
            except Exception as exc:
                print(f"  [skip] {stem}: {exc}")
        try:
            made += P.bbs_validation_curve(bbs_csv, os.path.join(fig, "bbs_validation_curve"),
                                           metric="val_comet", dpi=args.dpi)
        except Exception as exc:
            print(f"  [skip] bbs_validation_curve: {exc}")

    buckets = os.path.join(c, "bucket_results.csv")
    if _has(buckets):
        for metric in ("bleu", "comet"):
            try:
                made += H.config_bucket_heatmap(buckets, os.path.join(fig, f"config_bucket_{metric}_heatmap"),
                                                metric=metric, cmap=args.cmap, dpi=args.dpi)
            except Exception as exc:
                print(f"  [skip] config_bucket_{metric}_heatmap: {exc}")
        for metric, stem in (("bleu", "bleu_vs_length"), ("comet", "comet_vs_length")):
            try:
                made += P.metric_vs_length(buckets, os.path.join(fig, stem), metric=metric, dpi=args.dpi)
            except Exception as exc:
                print(f"  [skip] {stem}: {exc}")
        try:
            made += P.length_error_vs_length(buckets, os.path.join(fig, "length_error_vs_length"), dpi=args.dpi)
        except Exception as exc:
            print(f"  [skip] length_error_vs_length: {exc}")
        for fx, ft in (("A", "B"), ("A", "D"), ("B", "D"), ("C", "D")):
            try:
                made += P.interaction_plot(buckets, os.path.join(fig, f"interaction_{fx}x{ft}"),
                                           fx, ft, metric="bleu", dpi=args.dpi)
            except Exception as exc:
                print(f"  [skip] interaction_{fx}x{ft}: {exc}")

    fxcsv = os.path.join(c, "factorial_effects.csv")
    if _has(fxcsv):
        try:
            made += P.factorial_effects_plot(fxcsv, os.path.join(fig, "factorial_effects"), dpi=args.dpi)
        except Exception as exc:
            print(f"  [skip] factorial_effects: {exc}")

    v1v2 = os.path.join(c, "v1_v2_scale_ablation.csv")
    if _has(v1v2):
        try:
            from subprocess import run as _run

            _run([__import__("sys").executable,
                  os.path.join(os.path.dirname(__file__), "generate_v1_v2_heatmaps.py"),
                  "--csv", v1v2, "--fig-dir", fig], check=False)
            made.append(f"{fig}/v1_v2_scale_heatmap_*.{{png,pdf,svg}}")
        except Exception as exc:
            print(f"  [skip] v1_v2 heatmaps: {exc}")

    run_index = os.path.join(c, "run_index.csv")
    if _has(run_index):
        for metric in ("bleu", "comet", "chrf2"):
            try:
                made += P.seed_variation_plot(run_index, os.path.join(fig, f"seed_variation_{metric}"),
                                              metric=metric, dpi=args.dpi)
            except Exception as exc:
                print(f"  [skip] seed_variation_{metric}: {exc}")

    print("[figures] generated:")
    for p in made:
        print(f"  {p}")
    if not made:
        print("  (nothing — run evaluate_all.py / run_bbs_search.py / factorial_analysis.py first)")


if __name__ == "__main__":
    main()
