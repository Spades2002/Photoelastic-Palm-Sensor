"""Generate predicted-vs-ground-truth |Fz| plots for the three architectures with full per-sample predictions."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

from plot_style import apply_style, save_figure, DATA_DIR, force_magnitude, DATA_DIR, equal_axis_limits, DATA_DIR

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

    for display_name, path, kind, file_stub in MODELS:
        df = load_data(path, kind).dropna(subset=["Fz_true", "Fz_pred"]).copy()
        y_true = force_magnitude(df["Fz_true"])
        y_pred = force_magnitude(df["Fz_pred"])

        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        lo, hi = equal_axis_limits(y_true, y_pred)

        fig, ax = plt.subplots()
        ax.scatter(y_true, y_pred, s=34, alpha=0.8)
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, label="Ideal prediction")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"Ground-truth $|F_z|$ (N)")
        ax.set_ylabel(r"Predicted $|F_z|$ (N)")
        ax.set_title(f"{display_name}: Normal-Force Prediction")
        ax.text(
            0.04, 0.96,
            f"MAE = {mae:.3f} N\n$R^2$ = {r2:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )
        ax.legend(loc="lower right")
        save_figure(fig, f"fz_pred_vs_true_{file_stub}")
        plt.close(fig)



if __name__ == "__main__":
    main()
