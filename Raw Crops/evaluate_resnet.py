"""
Recomputes validation R2 directly from resnet_tactile_best.pt, the file
train_resnet.py actually saved to disk, rather than trusting the metrics
printed at the end of training. Those printed metrics come from whichever
epoch happened to be running when the loop ended, which is not always the
same epoch as the best checkpoint (that gets overwritten only when val loss
improves, so if it improved early and then crept up again, the saved file
is from an earlier epoch than the final printout describes).

Reproduces the exact same train/val point split train_resnet.py used, so
this evaluates on the same held-out points, not a different random split.
Uses whatever --test-size/--seed you actually trained with; the defaults
here match train_resnet.py's own defaults.

Usage:
    python evaluate_resnet.py
    python evaluate_resnet.py --test-size 0.25 --seed 0   # must match your training run
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

import config
from train_resnet import (
    TactileDataset, FORCE_TARGETS, LOCATION_TARGETS, _group_split, load_tactile_checkpoint,
)

logger = logging.getLogger(__name__)


def evaluate(test_size: float = 0.25, seed: int = 0, batch_size: int = 32, device_arg: str = "auto", num_workers: int = 4):
    ckpt_path = config.MODELS_ROOT / "resnet_tactile_best.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"{ckpt_path} not found, run train_resnet.py first")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_arg == "auto" else torch.device(device_arg)
    logger.info("Using device: %s", device)

    model, ckpt = load_tactile_checkpoint(ckpt_path, device)
    force_targets = ckpt.get("force_targets", FORCE_TARGETS)
    crop_source = ckpt.get("crop_source", "raw")
    logger.info("Checkpoint: force_targets=%s  crop_source=%s  force_conditions_location=%s  epoch=%s",
                force_targets, crop_source, ckpt.get("force_conditions_location", False), ckpt.get("epoch", "unknown"))
    force_mean, force_std = ckpt["force_mean"], ckpt["force_std"]
    loc_mean, loc_std = ckpt["loc_mean"], ckpt["loc_std"]

    df = pd.read_csv(config.DATASET_CSV_PATH)
    crop_cols = ["fringe_delta_path", "indent_delta_path"] if crop_source == "delta" else ["fringe_crop_path", "indent_crop_path"]
    df = df.dropna(subset=crop_cols + force_targets + LOCATION_TARGETS)
    _, val_df = _group_split(df, "point_id", test_size, seed)
    logger.info("Val points (%d): %s", val_df["point_id"].nunique(), sorted(val_df["point_id"].unique()))

    val_ds = TactileDataset(val_df, force_mean, force_std, loc_mean, loc_std, train=False,
                             force_targets=force_targets, crop_source=crop_source)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    all_force_pred, all_force_true, all_loc_pred, all_loc_true = [], [], [], []
    with torch.no_grad():
        for fringe, indent, force, loc in val_loader:
            fringe, indent = fringe.to(device), indent.to(device)
            force_pred, loc_pred = model(fringe, indent)
            all_force_pred.append(force_pred.cpu().numpy())
            all_force_true.append(force.numpy())
            all_loc_pred.append(loc_pred.cpu().numpy())
            all_loc_true.append(loc.numpy())

    force_pred = np.concatenate(all_force_pred) * force_std + force_mean
    force_true = np.concatenate(all_force_true) * force_std + force_mean
    loc_pred = np.concatenate(all_loc_pred) * loc_std + loc_mean
    loc_true = np.concatenate(all_loc_true) * loc_std + loc_mean

    metrics = {
        "force_r2": {t: float(r2_score(force_true[:, i], force_pred[:, i])) for i, t in enumerate(force_targets)},
        "location_r2": {t: float(r2_score(loc_true[:, i], loc_pred[:, i])) for i, t in enumerate(LOCATION_TARGETS)},
    }
    print(f"\nMetrics for the checkpoint actually on disk ({ckpt_path}):")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-size", type=float, default=0.25, help="Must match what train_resnet.py was run with")
    parser.add_argument("--seed", type=int, default=0, help="Must match what train_resnet.py was run with")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4, help="Set to 0 if you hit DataLoader/multiprocessing errors on Windows")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()
    evaluate(test_size=args.test_size, seed=args.seed, batch_size=args.batch_size, device_arg=args.device, num_workers=args.num_workers)
