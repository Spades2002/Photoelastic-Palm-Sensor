"""
Random-sample prediction sanity check.

Picks one row at random from each unique (row, col, cycle, depth_mm) group in
dataset.csv -- i.e. one sample per raw-image folder like
"Cycle_r10c8_5_down_5.000mm" -- restricted to the held-out test split, runs
the trained model on the matching pre-processed crop, and reports predicted
vs true depth and force.

This never reads from, writes to, or otherwise touches the raw image
folders. It only reads the CSV and the already-processed crops referenced in
it (read-only), and writes its outputs under config.OUTPUT_DIR.

Run with:
    python predict_random_samples.py
    python predict_random_samples.py --num-samples 40 --sample-seed 7
"""
from __future__ import annotations

import argparse
import logging
import random as random_module

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import random_split

import config
from dataset import TactileSensorDataset
from model import TactileForceNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Randomly sample test-set rows and check model accuracy.")
    parser.add_argument("--num-samples", type=int, default=25,
                         help="How many random (row, col, cycle, depth) groups to sample.")
    parser.add_argument("--sample-seed", type=int, default=None,
                         help="Seed for which groups/rows are picked. Default: a fresh random seed "
                              "every run (logged, so you can pass it back in to reproduce a specific draw).")
    parser.add_argument("--checkpoint", type=str, default=str(config.CHECKPOINT_DIR / "best_model.pt"),
                         help="Path to the model checkpoint to load.")
    return parser.parse_args()


def rebuild_test_split(full_dataset: TactileSensorDataset):
    """Recreate the exact same train/val/test split used in train.py, since
    random_split with the same seed and fractions over the same dataset is
    deterministic. This guarantees the samples checked here come only from
    data the model never trained or was tuned on."""
    n = len(full_dataset)
    n_val = int(n * config.VAL_SPLIT)
    n_test = int(n * config.TEST_SPLIT)
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(config.RANDOM_SEED)
    _, _, test_set = random_split(full_dataset, [n_train, n_val, n_test], generator=generator)
    return test_set


def pick_random_rows(full_frame: pd.DataFrame, test_positions: list, num_samples: int, sample_seed: int) -> pd.DataFrame:
    """From the test-split rows only, group by (row, col, cycle, depth_mm) --
    the CSV equivalent of one raw-image folder -- and pick one random row per
    group, up to num_samples groups.

    Each group gets its own derived seed (sample_seed + i), not one shared
    seed reused across every group. Reusing a single random_state for every
    .sample(n=1, ...) call would make pandas pick the same relative position
    within each group every time when groups are equal-sized, e.g. always
    the same phase index across every position, which isn't a genuine
    independent random draw."""
    test_frame = full_frame.iloc[test_positions]

    group_cols = [config.COL_ROW, config.COL_COL, config.COL_CYCLE, config.COL_DEPTH]
    missing = [c for c in group_cols if c not in test_frame.columns]
    if missing:
        raise ValueError(f"CSV is missing grouping columns needed for folder-style sampling: {missing}")

    grouped = test_frame.groupby(group_cols, dropna=False)
    group_keys = list(grouped.groups.keys())

    rng = random_module.Random(sample_seed)
    rng.shuffle(group_keys)
    chosen_keys = group_keys[:num_samples]

    chosen_rows = []
    for i, key in enumerate(chosen_keys):
        group_df = grouped.get_group(key)
        group_seed = (sample_seed + i) % (2**31 - 1)
        chosen_rows.append(group_df.sample(n=1, random_state=group_seed))
    return pd.concat(chosen_rows) if chosen_rows else test_frame.iloc[0:0]


def run_predictions(model, full_dataset: TactileSensorDataset, positions: list, device) -> list:
    model.eval()
    records = []
    with torch.no_grad():
        for pos in positions:
            sample = full_dataset[pos]
            fringe_image = sample["fringe_image"].unsqueeze(0).to(device)
            indent_image = sample["indent_image"].unsqueeze(0).to(device)

            outputs = model(fringe_image, indent_image)
            depth_pred = outputs["depth_pred"].item()
            xy_pred = outputs["xy_pred"].squeeze(0).cpu().numpy()
            force_pred = outputs["force_pred"].squeeze(0).cpu().numpy()

            depth_true = sample["depth_mm"].item()
            xy_true = sample["xy_mm"].numpy()
            force_true = sample["force"].numpy()
            meta = sample["meta"]

            records.append({
                "row": meta["row"], "col": meta["col"], "cycle": meta["cycle"], "phase": meta["phase"],
                "depth_mm_true": depth_true, "depth_mm_pred": depth_pred, "depth_mm_error": depth_pred - depth_true,
                "x_mm_true": xy_true[0], "x_mm_pred": xy_pred[0], "x_mm_error": xy_pred[0] - xy_true[0],
                "y_mm_true": xy_true[1], "y_mm_pred": xy_pred[1], "y_mm_error": xy_pred[1] - xy_true[1],
                "Fx_true": force_true[0], "Fx_pred": force_pred[0], "Fx_error": force_pred[0] - force_true[0],
                "Fy_true": force_true[1], "Fy_pred": force_pred[1], "Fy_error": force_pred[1] - force_true[1],
                "Fz_true": force_true[2], "Fz_pred": force_pred[2], "Fz_error": force_pred[2] - force_true[2],
                "magnitude_true": force_true[3], "magnitude_pred": force_pred[3],
                "magnitude_error": force_pred[3] - force_true[3],
                "_dataset_position": pos,
                "_fringe_image": sample["fringe_image"].squeeze(0).numpy(),
            })
    return records


def save_results_csv(records: list, path) -> None:
    rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in records]
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info("Saved random-sample predictions to %s", path)


def save_visual_grid(records: list, path) -> None:
    """A quick-look grid of the sampled fringe crops with predicted vs true
    depth and force magnitude in each title, so you can eyeball accuracy
    without opening the CSV."""
    n = len(records)
    if n == 0:
        logger.warning("No samples to visualise.")
        return

    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, record in zip(axes, records):
        ax.imshow(record["_fringe_image"], cmap="gray")
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

    full_dataset = TactileSensorDataset()
    test_set = rebuild_test_split(full_dataset)
    test_positions = list(test_set.indices)
    logger.info("Test split size: %d (sampling from this only)", len(test_positions))

    sample_seed = args.sample_seed
    if sample_seed is None:
        sample_seed = random_module.SystemRandom().randint(0, 2**31 - 1)
        logger.info(
            "No --sample-seed given; using a fresh random seed %d for this run "
            "(pass --sample-seed %d to reproduce this exact draw).",
            sample_seed, sample_seed,
        )

    sampled_frame = pick_random_rows(full_dataset.frame, test_positions, args.num_samples, sample_seed)
    sampled_positions = list(sampled_frame.index)
    logger.info("Sampled %d rows across distinct (row, col, cycle, depth) groups", len(sampled_positions))

    model = TactileForceNet(pretrained_backbone=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Loaded checkpoint from epoch %d (val_loss %.4f)", checkpoint["epoch"], checkpoint["val_loss"])

    records = run_predictions(model, full_dataset, sampled_positions, device)

    save_results_csv(records, config.PREDICTIONS_DIR / "random_sample_predictions.csv")
    save_visual_grid(records, config.PLOT_DIR / "random_sample_grid.png")


if __name__ == "__main__":
    main()
