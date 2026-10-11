"""
Line / bar plots (brief section 23, items 4-10).

All read from the aggregated CSVs; none hard-code numeric results.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from src.visualization.style import apply_pub_style, save

BUCKET_ORDER = ["B1", "B2", "B3", "B4", "B5"]
BUCKET_MID = {"B1": 3, "B2": 8, "B3": 13, "B4": 18, "B5": 24}


def metric_vs_length(
    bucket_csv: str,
    out_path_noext: str,
    metric: str = "bleu",
    configs: Optional[List[str]] = None,
    dpi: int = 300,
):
    import matplotlib.pyplot as plt

    apply_pub_style()
    df = pd.read_csv(bucket_csv)
    if metric not in df.columns:
        raise ValueError(f"{metric} not in {bucket_csv}")
    configs = configs or sorted(df["config"].unique())
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for c in configs:
        sub = df[df["config"] == c]
        g = sub.groupby("bucket")[metric].agg(["mean", "std"]).reindex(BUCKET_ORDER)
        x = [BUCKET_MID[b] for b in g.index]
        ax.errorbar(x, g["mean"], yerr=g["std"], marker="o", markersize=4,
                    capsize=2, linewidth=1.2, label=c)
    ax.set_xlabel("source length (approx. tokens, bucket midpoint)")
    ax.set_ylabel(f"{metric.upper()}" + ("" if metric != "comet" else " x100"))
    ax.set_title(f"{metric.upper()} vs source length")
    ax.legend(ncol=2, fontsize=7, frameon=False)
    return save(fig, out_path_noext, dpi=dpi)


def length_error_vs_length(bucket_csv: str, out_path_noext: str,
                           configs: Optional[List[str]] = None, dpi: int = 300):
    return metric_vs_length(bucket_csv, out_path_noext, metric="absolute_length_error",
                            configs=configs, dpi=dpi)


def factorial_effects_plot(effects_csv: str, out_path_noext: str, metric_label: str = "",
                           dpi: int = 300):
    """Horizontal bar of coefficients with 95% CI whiskers; * marks significance."""
    import matplotlib.pyplot as plt

    apply_pub_style()
    df = pd.read_csv(effects_csv)
    df = df[df["term"] != "Intercept"].copy()
    # main effects first, then 2-way, 3-way, 4-way
    df["order"] = df["term"].str.len()
    df = df.sort_values(["order", "term"])
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(6.0, 0.32 * len(df) + 1.5))
    err_low = df["coefficient"] - df["ci_low"]
    err_high = df["ci_high"] - df["coefficient"]
    ax.errorbar(df["coefficient"], y, xerr=[err_low, err_high], fmt="o",
                markersize=4, capsize=2, linewidth=1.1, color="#33608c")
    ax.axvline(0.0, color="0.4", linewidth=0.9, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels([
        t + (" *" if s else "") for t, s in zip(df["term"], df["significant"])
    ])
    ax.set_xlabel(f"regression coefficient{(' — ' + metric_label) if metric_label else ''}")
    ax.set_title("2^4 factorial effects (A=BBS B=TLC C=RoPE D=ALiBi)  * = p<0.05")
    ax.invert_yaxis()
    return save(fig, out_path_noext, dpi=dpi)


def interaction_plot(bucket_or_summary_csv: str, out_path_noext: str, factor_x: str,
                     factor_trace: str, metric: str = "bleu", dpi: int = 300):
    """Classic 2-factor interaction plot: mean `metric` vs factor_x, one line per factor_trace level."""
    import matplotlib.pyplot as plt

    from src.config import parse_case_id

    apply_pub_style()
    bit = {"A": "bbs", "B": "tlc", "C": "rope", "D": "alibi"}
    df = pd.read_csv(bucket_or_summary_csv)
    col = metric if metric in df.columns else f"{metric}_mean"
    if col not in df.columns:
        raise ValueError(f"{metric} not found in {bucket_or_summary_csv}")
    df = df.copy()
    df["fx"] = df["config"].map(lambda c: parse_case_id(c)[bit[factor_x]])
    df["ft"] = df["config"].map(lambda c: parse_case_id(c)[bit[factor_trace]])
    g = df.groupby(["ft", "fx"])[col].mean().reset_index()
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for lvl, sub in g.groupby("ft"):
        ax.plot(sub["fx"], sub[col], marker="o", label=f"{factor_trace}={lvl}")
    ax.set_xticks([0, 1])
    ax.set_xlabel(f"{factor_x} (0=off, 1=on)")
    ax.set_ylabel(f"mean {metric.upper()}")
    ax.set_title(f"{factor_x} x {factor_trace} interaction")
    ax.legend(frameon=False)
    return save(fig, out_path_noext, dpi=dpi)


def seed_variation_plot(run_index_csv: str, out_path_noext: str, metric: str = "bleu", dpi: int = 300):
    """Per-config scatter of individual seed scores + mean, to show seed spread."""
    import matplotlib.pyplot as plt

    apply_pub_style()
    df = pd.read_csv(run_index_csv)
    if metric not in df.columns:
        raise ValueError(f"{metric} not in {run_index_csv}")
    configs = sorted(df["config"].unique())
    fig, ax = plt.subplots(figsize=(max(6.0, 0.4 * len(configs) + 1), 4.0))
    for i, c in enumerate(configs):
        vals = df[df["config"] == c][metric].dropna().values
        ax.scatter([i] * len(vals), vals, s=18, color="#888", zorder=2)
        if len(vals):
            ax.scatter([i], [np.mean(vals)], marker="_", s=260, color="#c0392b", zorder=3)
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=90)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Seed variation — {metric.upper()} (grey = seed, red = mean)")
    return save(fig, out_path_noext, dpi=dpi)


def bbs_validation_curve(results_csv: str, out_path_noext: str, metric: str = "val_comet", dpi: int = 300):
    """Validation `metric` vs candidate id, manual & selected highlighted (brief section 23 item 9)."""
    import matplotlib.pyplot as plt

    apply_pub_style()
    df = pd.read_csv(results_csv).dropna(subset=[metric]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(range(len(df)), df[metric], marker="o", markersize=4, linewidth=1.0, color="#33608c")
    for tag, color in (("manual", "#c0392b"), ("uniform", "#e67e22"), ("selected", "#27ae60")):
        if "tag" in df.columns:
            m = df[df["tag"].astype(str).str.contains(tag, case=False, na=False)]
            for _, r in m.iterrows():
                ax.scatter([r.name], [r[metric]], s=70, color=color, zorder=3, label=tag)
    ax.set_xlabel("BBS weight candidate (ranked by search order)")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title("BBS validation performance vs weight configuration")
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        seen = dict(zip(labels, handles))
        ax.legend(seen.values(), seen.keys(), frameon=False)
    return save(fig, out_path_noext, dpi=dpi)
