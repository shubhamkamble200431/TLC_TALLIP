"""
Publication heatmaps (brief sections 12, 13, 23).

Visual style mirrors the reference figure the reviewers supplied:
  * numeric value written inside every cell,
  * a colorbar with a labelled quantity,
  * compact axis labels, serif typography, tight layout,
  * >= 300 DPI PNG plus a vector PDF,
  * colormap chosen through configuration (never hard-coded).

Everything is reproducible from the saved CSVs.
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from src.visualization.style import apply_pub_style, save


def annotated_heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    out_path_noext: str,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    cbar_label: str = "",
    cmap: str = "viridis",
    value_fmt: str = "{:.2f}",
    annot_matrix: Optional[np.ndarray] = None,
    dpi: int = 300,
    figsize=None,
    formats=("png", "pdf"),
):
    import matplotlib.pyplot as plt

    apply_pub_style()
    m = np.asarray(matrix, dtype=float)
    nrows, ncols = m.shape
    figsize = figsize or (max(4.0, 0.75 * ncols + 2.2), max(3.0, 0.6 * nrows + 1.8))
    fig, ax = plt.subplots(figsize=figsize)

    finite = m[np.isfinite(m)]
    vmin = float(finite.min()) if finite.size else 0.0
    vmax = float(finite.max()) if finite.size else 1.0
    im = ax.imshow(m, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_xticks(range(ncols))
    ax.set_xticklabels(col_labels, rotation=0)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)

    # cell annotations with contrast-aware text colour
    rng = (vmax - vmin) or 1.0
    for i in range(nrows):
        for j in range(ncols):
            v = m[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "--", ha="center", va="center", fontsize=8, color="0.5")
                continue
            txt = value_fmt.format(v) if annot_matrix is None else str(annot_matrix[i, j])
            frac = (v - vmin) / rng
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if frac < 0.55 else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label)
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    return save(fig, out_path_noext, formats=formats, dpi=dpi)


# ---------------------------------------------------------------------------
def bbs_weight_heatmap(
    results_csv: str,
    out_path_noext: str,
    metric: str = "val_comet",
    cmap: str = "viridis",
    dpi: int = 300,
    w_row: str = "normalized_w4",
    w_col: str = "normalized_w5",
):
    """
    Rows = w4 emphasis, Columns = w5 emphasis, cell = validation `metric`
    (brief section 12).  Reproduced entirely from bbs_search_results.csv.
    """
    df = pd.read_csv(results_csv)
    if metric not in df.columns:
        raise ValueError(f"{metric} not in {results_csv}; have {list(df.columns)}")
    df = df.dropna(subset=[metric])
    piv = df.pivot_table(index=w_row, columns=w_col, values=metric, aggfunc="mean")
    piv = piv.sort_index(ascending=True).sort_index(axis=1, ascending=True)
    lower_better = metric.endswith("length_error") or metric.endswith("ter")
    cbar = metric.replace("_", " ") + (" (lower better)" if lower_better else "")
    return annotated_heatmap(
        piv.values,
        [f"{x:.2f}" for x in piv.index],
        [f"{x:.2f}" for x in piv.columns],
        out_path_noext,
        title=f"BBS weight search — {metric.replace('_', ' ')}",
        xlabel="relative emphasis on B5 (w5, mean-normalised)",
        ylabel="relative emphasis on B4 (w4, mean-normalised)",
        cbar_label=cbar,
        cmap=cmap,
        value_fmt="{:.2f}",
        dpi=dpi,
    )


def config_bucket_heatmap(
    summary_bucket_csv: str,
    out_path_noext: str,
    metric: str = "bleu",
    cmap: str = "viridis",
    dpi: int = 300,
    annotate_std: bool = True,
    config_order: Optional[List[str]] = None,
):
    """
    Rows = X-ABCD configs, Columns = B1..B5, cell = 3-seed mean of `metric`
    (brief section 13).  Cell text = "mean" or "mean±std".
    """
    df = pd.read_csv(summary_bucket_csv)
    if metric not in df.columns:
        raise ValueError(f"{metric} not in {summary_bucket_csv}")
    g = df.groupby(["config", "bucket"])[metric].agg(["mean", "std"]).reset_index()
    configs = config_order or sorted(g["config"].unique())
    buckets = ["B1", "B2", "B3", "B4", "B5"]
    mean_m = np.full((len(configs), len(buckets)), np.nan)
    annot = np.empty((len(configs), len(buckets)), dtype=object)
    for i, c in enumerate(configs):
        for j, b in enumerate(buckets):
            sub = g[(g["config"] == c) & (g["bucket"] == b)]
            if sub.empty:
                annot[i, j] = "--"
                continue
            mu = float(sub["mean"].iloc[0])
            sd = sub["std"].iloc[0]
            mean_m[i, j] = mu
            if annotate_std and sd == sd:
                annot[i, j] = f"{mu:.1f}\n±{float(sd):.1f}"
            else:
                annot[i, j] = f"{mu:.1f}"
    return annotated_heatmap(
        mean_m, configs, buckets, out_path_noext,
        title=f"Configuration x length bucket — mean {metric.upper()}",
        xlabel="source-length bucket", ylabel="configuration (X-ABCD)",
        cbar_label=f"mean {metric.upper()} (3 seeds)",
        cmap=cmap, annot_matrix=annot, dpi=dpi,
        figsize=(7.5, 0.42 * len(configs) + 1.8),
    )
