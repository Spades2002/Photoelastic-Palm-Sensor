"""Generate absolute |Fz| error distributions for each architecture with full per-sample predictions."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from plot_style import apply_style, save_figure, DATA_DIR, force_magnitude, DATA_DIR

BASE = DATA_DIR

MODELS = [
    ("Joint Multi-Task", BASE / "joint_multitask_predictions.xlsx", "excel", "joint_multitask"),
    ("Force-Conditioned", BASE / "force_conditioned_predictions.xlsx", "excel", "force_conditioned"),
    ("Physics-Informed Two-Stage", BASE / "two_stage_predictions.csv", "csv", "two_stage"),
]


def load_data(path, kind):
    return pd.read_excel(path) if kind == "excel" else pd.read_csv(path)


def main():
    apply_style()

    for name, path, kind, stub in MODELS:
        df = load_data(path, kind).dropna(subset=["Fz_true", "Fz_pred"]).copy()
        y_true = force_magnitude(df["Fz_true"])
        y_pred = force_magnitude(df["Fz_pred"])
        abs_error = np.abs(y_true - y_pred)

        fig, ax = plt.subplots()
        bins = max(6, min(12, int(np.sqrt(len(abs_error))) + 2))
        ax.hist(abs_error, bins=bins, edgecolor="black", alpha=0.8)
        ax.axvline(abs_error.mean(), linestyle="--", linewidth=1.2, label=f"Mean = {abs_error.mean():.3f} N")
        ax.set_xlabel(r"Absolute $F_z$ error (N)")
        ax.set_ylabel("Number of samples")
        ax.set_title(f"{name}: Normal-Force Absolute-Error Distribution")
        ax.set_xlim(left=0)
        ax.legend(loc="best")
        save_figure(fig, f"fz_absolute_error_distribution_{stub}")
        plt.close(fig)



if __name__ == "__main__":
    main()
