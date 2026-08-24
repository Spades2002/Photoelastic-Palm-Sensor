"""Generate 2D ground-truth vs predicted contact-location plots for the CNN models."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from plot_style import apply_style, save_figure, DATA_DIR

BASE = DATA_DIR

MODELS = [
    ("Joint Multi-Task", BASE / "joint_multitask_predictions.xlsx", "excel", "joint_multitask", "x_mm_true", "x_mm_pred", "y_mm_true", "y_mm_pred"),
    ("Force-Conditioned", BASE / "force_conditioned_predictions.xlsx", "excel", "force_conditioned", "x_mm_true", "x_mm_pred", "y_mm_true", "y_mm_pred"),
    ("Physics-Informed Two-Stage", BASE / "two_stage_predictions.csv", "csv", "two_stage", "x_mm_true", "x_mm_pred", "y_mm_true", "y_mm_pred"),
]


def load_data(path, kind):
    return pd.read_excel(path) if kind == "excel" else pd.read_csv(path)


def main():
    apply_style()

    for name, path, kind, stub, xt, xp, yt, yp in MODELS:
        df = load_data(path, kind).dropna(subset=[xt, xp, yt, yp]).copy()
        x_true = df[xt].to_numpy(dtype=float)
        x_pred = df[xp].to_numpy(dtype=float)
        y_true = df[yt].to_numpy(dtype=float)
        y_pred = df[yp].to_numpy(dtype=float)
        errors = np.sqrt((x_true - x_pred) ** 2 + (y_true - y_pred) ** 2)

        fig, ax = plt.subplots(figsize=(5.4, 5.0))
        ax.scatter(x_true, y_true, s=42, marker="o", label="Ground truth")
        ax.scatter(x_pred, y_pred, s=42, marker="x", label="Prediction")

        for x0, y0, x1, y1 in zip(x_true, y_true, x_pred, y_pred):
            ax.plot([x0, x1], [y0, y1], linewidth=0.7, alpha=0.5)

        all_x = np.concatenate([x_true, x_pred])
        all_y = np.concatenate([y_true, y_pred])
        pad = 2.0
        ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
        ax.set_ylim(all_y.min() - pad, all_y.max() + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Contact position, x (mm)")
        ax.set_ylabel("Contact position, y (mm)")
        ax.set_title(f"{name}: Contact Localisation")
        ax.text(
            0.04, 0.96,
            f"Mean Euclidean error = {errors.mean():.3f} mm",
            transform=ax.transAxes,
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )
        ax.legend(loc="best")
        save_figure(fig, f"contact_localisation_{stub}")
        plt.close(fig)


if __name__ == "__main__":
    main()
