"""Publication-quality matplotlib defaults (brief sections 12, 23)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PUB_RC = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman", "serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": False,
}


def apply_pub_style():
    plt.rcParams.update(PUB_RC)


def save(fig, out_path_noext: str, formats=("png", "pdf"), dpi: int = 300):
    """Save `fig` to <out_path_noext>.<fmt> for each fmt; returns the paths."""
    import os

    os.makedirs(os.path.dirname(os.path.abspath(out_path_noext)), exist_ok=True)
    paths = []
    for fmt in formats:
        p = f"{out_path_noext}.{fmt}"
        fig.savefig(p, format=fmt, dpi=dpi, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths
