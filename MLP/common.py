"""
Shared configuration and helpers. Every other script in this folder
imports from here, so edit CONFIG once and it applies everywhere.
"""
from __future__ import annotations

import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CONFIG = {
    "root_dir": r"D:\ERP\Pictures for Dataset",            # holds the per-cycle CSVs
    "images_dir": r"D:\ERP\Pictures for Dataset\images",   # kept for reference only
    "active_roi_frac": (0.21, 0.09, 0.9, 0.80),
    "fringe_roi_frac": (0.0, 0.0, 1.0, 0.42),
    "flatten_size": (64, 64),
    "val_fraction": 0.15,
    "seed": 42,
    "output_dir": r"D:\ERP\Data Analysis 4",
}

FORCE_COLUMNS = ["Fx", "Fy", "Fz"]


def features_path(mode: str) -> str:
    return os.path.join(CONFIG["output_dir"], f"features_{mode}.npz")


def load_features(mode: str) -> tuple:
    """Loads the cached feature arrays written by build_features.py."""
    path = features_path(mode)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run build_features.py first, it only needs to run once, "
            f"every train_*.py script reads from its cached output."
        )
    data = np.load(path)
    return data["X_train"], data["X_val"], data["y_train"], data["y_val"]


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    metrics = {}
    for i, axis in enumerate(FORCE_COLUMNS):
        metrics[f"{axis}_mae"] = mean_absolute_error(y_true[:, i], y_pred[:, i])
        metrics[f"{axis}_rmse"] = mean_squared_error(y_true[:, i], y_pred[:, i]) ** 0.5
        metrics[f"{axis}_r2"] = r2_score(y_true[:, i], y_pred[:, i])
    metrics["overall_mae"] = mean_absolute_error(y_true, y_pred)
    metrics["overall_r2"] = r2_score(y_true, y_pred)
    return metrics


def run_model(name: str, model) -> pd.DataFrame:
    """Trains `model` on both the raw and delta feature caches, evaluates each,
    and saves this model's own results_<name>.csv plus fitted .joblib files.
    Writing a model-specific results file, rather than one shared file, means
    these scripts can be run independently, including at the same time in
    separate terminals, without clashing over the same file."""
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    rows = []
    for mode in ("raw", "delta"):
        X_train, X_val, y_train, y_val = load_features(mode)
        print(f"[{name}] mode={mode}: train={X_train.shape[0]}, val={X_val.shape[0]}, "
              f"features={X_train.shape[1]}")

        start = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        elapsed = time.time() - start

        metrics = evaluate(y_val, y_pred)
        metrics["model"] = name
        metrics["feature_mode"] = mode
        metrics["train_seconds"] = round(elapsed, 2)
        rows.append(metrics)
        print(f"[{name}] mode={mode}: overall MAE {metrics['overall_mae']:.4f}, "
              f"overall R2 {metrics['overall_r2']:.4f}, {elapsed:.1f}s")

        joblib.dump(model, os.path.join(CONFIG["output_dir"], f"{name}_{mode}.joblib"))

    results = pd.DataFrame(rows)
    results_path = os.path.join(CONFIG["output_dir"], f"results_{name}.csv")
    results.to_csv(results_path, index=False)
    print(f"[{name}] results written to {results_path}")
    return results
