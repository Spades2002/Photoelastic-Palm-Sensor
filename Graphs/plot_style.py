"""Shared plotting utilities for the thesis figures."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path(r"D:\\ERP\\Graphs")
OUTPUT_DIR = DATA_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def apply_style():
    """Apply a consistent, publication-friendly Matplotlib style."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.figsize": (5.8, 4.2),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
    })


def save_figure(fig, filename):
    """Save both PNG and PDF versions of a figure."""
    stem = Path(filename).stem
    fig.savefig(OUTPUT_DIR / f"{stem}.png")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf")
    print(f"Saved: {OUTPUT_DIR / (stem + '.png')}")
    print(f"Saved: {OUTPUT_DIR / (stem + '.pdf')}")


def force_magnitude(series):
    """Convert signed compressive Fz values to force magnitude."""
    return np.abs(np.asarray(series, dtype=float))


def equal_axis_limits(true_values, pred_values, padding=0.05):
    """Return common axis limits for predicted-vs-ground-truth plots."""
    values = np.concatenate([np.asarray(true_values), np.asarray(pred_values)])
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    span = hi - lo
    pad = padding * span if span > 0 else 0.5
    return lo - pad, hi + pad
