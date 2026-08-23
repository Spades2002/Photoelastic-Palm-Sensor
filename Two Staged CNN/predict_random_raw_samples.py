r"""
Like predict_random_samples.py, but through the RAW-PHOTO pipeline instead
of the pre-processed crops: for each randomly picked test-split shot, this
reconstructs the actual raw photo + baseline photo paths under
config.RAW_IMAGES_ROOT, crops the fringe/indent ROIs, computes the delta,
runs the model, and compares against the true values already in
dataset.csv for that row.

This is the multi-sample version of what predict_visual.py does for one
shot, useful for confirming the raw-photo pipeline (ROI + delta formula)
holds up across many samples, not just the one we hand-verified.

Never writes into RAW_IMAGES_ROOT, CSV_PATH, or IMAGE_ROOT. Only reads them.

Run with:
    python predict_random_raw_samples.py
    python predict_random_raw_samples.py --num-samples 40 --sample-seed 7
"""
from __future__ import annotations

import argparse
import logging
import random as random_module
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split

import config
from cv_utils import compute_delta_crop, crop_fractional_roi
from dataset import TactileSensorDataset
from model import TactileForceNet
from predict_visual import find_baseline_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Randomly sample raw photos and run the full raw-photo pipeline.")
    parser.add_argument("--num-samples", type=int, default=25,
                         help="How many random (row, col, cycle, depth) groups to sample.")
    parser.add_argument("--sample-seed", type=int, default=None,
                         help="Seed for which groups/rows are picked. Default: a fresh random seed every run.")
    parser.add_argument("--checkpoint", type=str, default=str(config.CHECKPOINT_DIR / "best_model.pt"))
    return parser.parse_args()


def rebuild_test_frame() -> pd.DataFrame:
    """Load dataset.csv and restrict to the same held-out test split
    train.py used (same seed, same fractions), so sampled rows were never
    trained or tuned on."""
    full_dataset = TactileSensorDataset()
    n = len(full_dataset)
    n_val = int(n * config.VAL_SPLIT)
    n_test = int(n * config.TEST_SPLIT)
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    _, _, test_set = random_split(full_dataset, [n_train, n_val, n_test], generator=generator)
    test_positions = list(test_set.indices)
    return full_dataset.frame.iloc[test_positions]


def raw_image_path(row: pd.Series) -> Path:
    folder = f"Cycle_r{int(row[config.COL_ROW])}c{int(row[config.COL_COL])}_{int(row[config.COL_CYCLE])}_" \
             f"{row[config.COL_PHASE]}_{float(row[config.COL_DEPTH]):.3f}mm"
    filename = f"{int(row[config.COL_IMG_IDX]):02d}.jpg"
    return config.RAW_IMAGES_ROOT / folder / filename


def pick_random_rows(test_frame: pd.DataFrame, num_samples: int, sample_seed: int) -> list:
    """Group by (row, col, cycle, depth_mm), pick one random row per group,
    skipping any group whose raw photo or baseline photo can't be found on
    disk, until num_samples valid rows are collected or groups run out."""
    group_cols = [config.COL_ROW, config.COL_COL, config.COL_CYCLE, config.COL_DEPTH]
    grouped = test_frame.groupby(group_cols, dropna=False)
    group_keys = list(grouped.groups.keys())

    rng = random_module.Random(sample_seed)
    rng.shuffle(group_keys)

    chosen = []
    skipped = 0
    for i, key in enumerate(group_keys):
        if len(chosen) >= num_samples:
            break
        group_df = grouped.get_group(key)
        group_seed = (sample_seed + i) % (2**31 - 1)
        row = group_df.sample(n=1, random_state=group_seed).iloc[0]

        img_path = raw_image_path(row)
        if not img_path.exists():
            skipped += 1
            continue
        baseline_path = find_baseline_image(img_path)
        if baseline_path is None or not baseline_path.exists():
            skipped += 1
            continue

        chosen.append((row, img_path, baseline_path))

    if skipped:
        logger.info("Skipped %d group(s) whose raw photo or baseline couldn't be found on disk.", skipped)
    return chosen


def preprocess_crop(gray_delta: np.ndarray) -> torch.Tensor:
    resized = cv2.resize(gray_delta, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    return torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0


def run_predictions(model, chosen: list, device) -> list:
    records = []
    with torch.no_grad():
        for row, img_path, baseline_path in chosen:
            raw = cv2.imread(str(img_path))
            baseline_raw = cv2.imread(str(baseline_path))
            if raw is None or baseline_raw is None:
                logger.warning("Could not read %s or its baseline; skipping.", img_path)
                continue

            fringe_current = cv2.cvtColor(crop_fractional_roi(raw, config.FRINGE_ROI_FRAC), cv2.COLOR_BGR2GRAY)
            fringe_baseline = cv2.cvtColor(crop_fractional_roi(baseline_raw, config.FRINGE_ROI_FRAC), cv2.COLOR_BGR2GRAY)
            indent_current = cv2.cvtColor(crop_fractional_roi(raw, config.INDENT_ROI_FRAC), cv2.COLOR_BGR2GRAY)
            indent_baseline = cv2.cvtColor(crop_fractional_roi(baseline_raw, config.INDENT_ROI_FRAC), cv2.COLOR_BGR2GRAY)

            fringe_delta = compute_delta_crop(fringe_current, fringe_baseline)
            indent_delta = compute_delta_crop(indent_current, indent_baseline)

            fringe_tensor = preprocess_crop(fringe_delta).to(device)
            indent_tensor = preprocess_crop(indent_delta).to(device)
            outputs = model(fringe_tensor, indent_tensor)

            depth_pred = outputs["depth_pred"].item()
            xy_pred = outputs["xy_pred"].squeeze(0).cpu().numpy()
            force_pred = outputs["force_pred"].squeeze(0).cpu().numpy()

            depth_true = float(row[config.COL_DEPTH])
            x_true = float(row[config.COL_X])
            y_true = float(row[config.COL_Y])
            fx_true = float(row[config.COL_FX])
            fy_true = float(row[config.COL_FY])
            fz_true = float(row[config.COL_FZ])
            mag_true = float(np.sqrt(fx_true**2 + fy_true**2 + fz_true**2))

            records.append({
                "row": int(row[config.COL_ROW]), "col": int(row[config.COL_COL]),
                "cycle": int(row[config.COL_CYCLE]), "phase": row[config.COL_PHASE],
                "depth_mm_true": depth_true, "depth_mm_pred": depth_pred,
                "depth_mm_error": depth_pred - depth_true,
                "x_mm_true": x_true, "x_mm_pred": float(xy_pred[0]), "x_mm_error": float(xy_pred[0]) - x_true,
                "y_mm_true": y_true, "y_mm_pred": float(xy_pred[1]), "y_mm_error": float(xy_pred[1]) - y_true,
                "Fx_true": fx_true, "Fx_pred": float(force_pred[0]), "Fx_error": float(force_pred[0]) - fx_true,
                "Fy_true": fy_true, "Fy_pred": float(force_pred[1]), "Fy_error": float(force_pred[1]) - fy_true,
                "Fz_true": fz_true, "Fz_pred": float(force_pred[2]), "Fz_error": float(force_pred[2]) - fz_true,
                "magnitude_true": mag_true, "magnitude_pred": float(force_pred[3]),
                "magnitude_error": float(force_pred[3]) - mag_true,
                "_fringe_image": fringe_delta,
            })
    return records


def save_results_csv(records: list, path: Path) -> None:
    rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in records]
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info("Saved raw-photo random-sample predictions to %s", path)


def save_visual_grid(records: list, path: Path) -> None:
    n = len(records)
    if n == 0:
        logger.warning("No samples to visualise.")
        return

    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, record in zip(axes, records):
        resized = cv2.resize(record["_fringe_image"], (config.IMAGE_SIZE, config.IMAGE_SIZE))
        ax.imshow(resized, cmap="gray")
        ax.set_title(
            f"r{record['row']}c{record['col']} cyc{record['cycle']}\n"
            f"depth {record['depth_mm_true']:.2f}->{record['depth_mm_pred']:.2f}\n"
            f"xy ({record['x_mm_true']:.1f},{record['y_mm_true']:.1f})->"
            f"({record['x_mm_pred']:.1f},{record['y_mm_pred']:.1f})\n"
            f"|F| {record['magnitude_true']:.2f}->{record['magnitude_pred']:.2f}",
            fontsize=7,
        )
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved visual sample grid to %s", path)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    sample_seed = args.sample_seed
    if sample_seed is None:
        sample_seed = random_module.SystemRandom().randint(0, 2**31 - 1)
        logger.info("No --sample-seed given; using a fresh random seed %d for this run "
                     "(pass --sample-seed %d to reproduce this exact draw).", sample_seed, sample_seed)

    test_frame = rebuild_test_frame()
    logger.info("Test split size: %d (sampling from this only)", len(test_frame))

    chosen = pick_random_rows(test_frame, args.num_samples, sample_seed)
    logger.info("Found raw photos + baselines for %d/%d requested samples", len(chosen), args.num_samples)

    model = TactileForceNet(pretrained_backbone=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info("Loaded checkpoint from epoch %d (val_loss %.4f)", checkpoint["epoch"], checkpoint["val_loss"])

    records = run_predictions(model, chosen, device)

    save_results_csv(records, config.PREDICTIONS_DIR / "random_raw_sample_predictions.csv")
    save_visual_grid(records, config.PLOT_DIR / "random_raw_sample_grid.png")


if __name__ == "__main__":
    main()
