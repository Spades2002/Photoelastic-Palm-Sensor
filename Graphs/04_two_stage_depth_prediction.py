"""Generate predicted-vs-ground-truth indentation depth for the two-stage pipeline."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

from plot_style import apply_style, save_figure, DATA_DIR, equal_axis_limits, DATA_DIR

BASE = DATA_DIR


def main():
    apply_style()
    df = pd.read_csv(BASE / "two_stage_predictions.csv")

    y_true = df["depth_mm_true"].to_numpy(dtype=float)
    y_pred = df["depth_mm_pred"].to_numpy(dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    lo, hi = equal_axis_limits(y_true, y_pred)

    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, s=36, alpha=0.8)
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, label="Ideal prediction")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Ground-truth indentation depth (mm)")
    ax.set_ylabel("Predicted indentation depth (mm)")
    ax.set_title("Physics-Informed Two-Stage: Indentation-Depth Prediction")
    ax.text(
        0.04, 0.96,
        f"MAE = {mae:.3f} mm\n$R^2$ = {r2:.3f}",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )
    ax.legend(loc="lower right")
    save_figure(fig, "two_stage_depth_pred_vs_true")
    plt.close(fig)


if __name__ == "__main__":
    main()
