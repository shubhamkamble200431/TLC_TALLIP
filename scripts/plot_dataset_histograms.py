#!/usr/bin/env python
"""
Source-length bucket histograms for the V1 and V2 corpora.

Buckets (source side, whitespace tokens):
    B1 = 1-5   B2 = 6-10   B3 = 11-15   B4 = 16-20   B5 = 21+

Produces (>=300 DPI PNG + PDF), reproducible from the corpora on disk:

    results/figures/dataset_bucket_hist_counts.{png,pdf}    grouped bars, counts
    results/figures/dataset_bucket_hist_share.{png,pdf}     grouped bars, % of split
    results/figures/dataset_bucket_hist_train.{png,pdf}     V1 vs V2 train, counts + %
    results/json/dataset_bucket_histogram.json              the exact numbers

Also prints the count table to stdout.

Usage:
    python scripts/plot_dataset_histograms.py
    python scripts/plot_dataset_histograms.py --splits train val test_sampled
"""
import _paths  # noqa: F401

import argparse
import os

import numpy as np

from src.data.buckets import DEFAULT_BOUNDARIES, assign_bucket, n_tokens
from src.utils.io import ensure_dir, read_lines, write_json
from src.visualization.style import apply_pub_style, save

BUCKET_LABELS = ["B1\n(1-5)", "B2\n(6-10)", "B3\n(11-15)", "B4\n(16-20)", "B5\n(21+)"]
V1_COLOR = "#4C72B0"
V2_COLOR = "#DD8452"


def _bucket_counts(path, boundaries=DEFAULT_BOUNDARIES):
    if not os.path.isfile(path):
        return None
    c = np.zeros(5, dtype=int)
    for ln in read_lines(path):
        c[assign_bucket(n_tokens(ln), boundaries) - 1] += 1
    return c


def _corpus_splits(data_dir, version):
    sampled = "test_7264_5" if version == "v2" else "test_5414_5"
    return {
        "train": os.path.join(data_dir, "train.sa"),
        "val": os.path.join(data_dir, "val.sa"),
        "test_unsampled": os.path.join(data_dir, "test.sa"),
        "test_sampled": os.path.join(data_dir, f"{sampled}.sa"),
    }


def _grouped_bar(ax, v1, v2, title, ylabel, as_pct=False, annotate=True):
    x = np.arange(5)
    w = 0.38
    a = v1.astype(float)
    b = v2.astype(float)
    if as_pct:
        a = 100 * a / max(a.sum(), 1)
        b = 100 * b / max(b.sum(), 1)
    r1 = ax.bar(x - w / 2, a, w, label="V1", color=V1_COLOR, edgecolor="white", linewidth=0.4)
    r2 = ax.bar(x + w / 2, b, w, label="V2", color=V2_COLOR, edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(BUCKET_LABELS)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("source-length bucket")
    top = max(a.max(), b.max())
    ax.set_ylim(0, top * (1.20 if annotate else 1.05))
    if annotate:
        fmt = (lambda v: f"{v:.1f}%") if as_pct else (lambda v: f"{int(round(v)):,}")
        for rects, vals in ((r1, a), (r2, b)):
            for rect, v in zip(rects, vals):
                ax.text(rect.get_x() + rect.get_width() / 2, v + top * 0.015, fmt(v),
                        ha="center", va="bottom", fontsize=7, rotation=90 if not as_pct else 0)
    ax.legend(frameon=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v1-dir", default="data/v1")
    ap.add_argument("--v2-dir", default="data/v2")
    ap.add_argument("--splits", nargs="*",
                    default=["train", "val", "test_sampled", "test_unsampled"])
    ap.add_argument("--fig-dir", default="results/figures")
    ap.add_argument("--json-out", default="results/json/dataset_bucket_histogram.json")
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    import matplotlib.pyplot as plt

    apply_pub_style()

    v1p = _corpus_splits(args.v1_dir, "v1")
    v2p = _corpus_splits(args.v2_dir, "v2")
    data = {"buckets": ["B1(1-5)", "B2(6-10)", "B3(11-15)", "B4(16-20)", "B5(21+)"],
            "V1": {}, "V2": {}}
    counts = {"V1": {}, "V2": {}}
    for split in args.splits:
        for name, paths, store in (("V1", v1p, counts["V1"]), ("V2", v2p, counts["V2"])):
            c = _bucket_counts(paths[split])
            if c is None:
                continue
            store[split] = c
            data[name][split] = {"counts": c.tolist(), "total": int(c.sum()),
                                 "share_pct": [round(x, 3) for x in (100 * c / max(c.sum(), 1))]}

    ensure_dir(os.path.dirname(os.path.abspath(args.json_out)))
    write_json(args.json_out, data)

    # ---- stdout table -------------------------------------------------
    print(f"\n{'corpus/split':<20} {'B1':>9} {'B2':>9} {'B3':>9} {'B4':>9} {'B5':>9} {'total':>10}")
    print("-" * 80)
    for name in ("V1", "V2"):
        for split in args.splits:
            if split not in counts[name]:
                continue
            c = counts[name][split]
            print(f"{name + ' ' + split:<20} " + " ".join(f"{int(v):>9,}" for v in c) + f" {int(c.sum()):>10,}")
    print()

    made = []
    present = [s for s in args.splits if s in counts["V1"] and s in counts["V2"]]

    # ---- Fig 1: grouped bars, COUNTS, one panel per split -----------
    fig, axes = plt.subplots(1, len(present), figsize=(4.6 * len(present), 4.4), squeeze=False)
    for ax, split in zip(axes[0], present):
        _grouped_bar(ax, counts["V1"][split], counts["V2"][split],
                     title=split.replace("_", " "), ylabel="sentence count")
    fig.suptitle("Source-length bucket distribution — V1 vs V2 (counts)", y=1.02, fontweight="bold")
    made += save(fig, os.path.join(args.fig_dir, "dataset_bucket_hist_counts"), dpi=args.dpi)

    # ---- Fig 2: grouped bars, SHARE % (shape comparison) -----------
    fig, axes = plt.subplots(1, len(present), figsize=(4.6 * len(present), 4.4), squeeze=False)
    for ax, split in zip(axes[0], present):
        _grouped_bar(ax, counts["V1"][split], counts["V2"][split],
                     title=split.replace("_", " "), ylabel="% of split", as_pct=True)
    fig.suptitle("Source-length bucket distribution — V1 vs V2 (share of split)", y=1.02, fontweight="bold")
    made += save(fig, os.path.join(args.fig_dir, "dataset_bucket_hist_share"), dpi=args.dpi)

    # ---- Fig 3: TRAIN only, counts + % side by side ---------------
    if "train" in present:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        _grouped_bar(axes[0], counts["V1"]["train"], counts["V2"]["train"],
                     title="train — counts", ylabel="sentence count")
        _grouped_bar(axes[1], counts["V1"]["train"], counts["V2"]["train"],
                     title="train — share", ylabel="% of train", as_pct=True)
        fig.suptitle("Training-set source-length buckets — V1 vs V2", y=1.02, fontweight="bold")
        made += save(fig, os.path.join(args.fig_dir, "dataset_bucket_hist_train"), dpi=args.dpi)

    print("[dataset histograms] generated:")
    for p in made:
        print(f"  {p}")
    print(f"  {args.json_out}")


if __name__ == "__main__":
    main()
