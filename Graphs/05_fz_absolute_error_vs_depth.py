"""Generate |Fz| absolute-error vs indentation-depth plots for the three architectures with full per-sample predictions."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from plot_style import apply_style, save_figure, DATA_DIR, force_magnitude, DATA_DIR

BASE = DATA_DIR

MODELS = [
    ("Joint Multi-Task", BASE / "joint_multitask_predictions.xlsx", "excel", "depth_mm", "joint_multitask"),
    ("Force-Conditioned", BASE / "force_conditioned_predictions.xlsx", "excel", "depth_mm", "force_conditioned"),
    ("Physics-Informed Two-Stage", BASE / "two_stage_predictions.csv", "csv", "depth_mm_true", "two_stage"),
]


def load_data(path, kind):
    return pd.read_excel(path) if kind == "excel" else pd.read_csv(path)


def main():
    apply_style()

    for name, path, kind, depth_col, stub in MODELS:
        df = load_data(path, kind).dropna(subset=[depth_col, "Fz_true", "Fz_pred"]).copy()
        depth = df[depth_col].to_numpy(dtype=float)
        true_mag = force_magnitude(df["Fz_true"])
        pred_mag = force_magnitude(df["Fz_pred"])
        abs_error = np.abs(true_mag - pred_mag)

        fig, ax = plt.subplots()
        ax.scatter(depth, abs_error, s=34, alpha=0.8)

        # Add a simple least-squares trend line where enough samples exist.
        if len(depth) >= 3 and np.ptp(depth) > 0:
            coeff = np.polyfit(depth, abs_error, 1)
            x_line = np.linspace(depth.min(), depth.max(), 100)
            y_line = np.polyval(coeff, x_line)
            ax.plot(x_line, y_line, linestyle="--", linewidth=1.2, label="Linear trend")
            ax.legend(loc="best")

        ax.set_xlabel("Indentation depth (mm)")
        ax.set_ylabel(r"Absolute $F_z$ error (N)")
        ax.set_title(f"{name}: Normal-Force Error vs Indentation Depth")
        ax.set_ylim(bottom=0)
        save_figure(fig, f"fz_absolute_error_vs_depth_{stub}")
        plt.close(fig)



if __name__ == "__main__":
    main()
