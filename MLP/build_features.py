"""
Run this once before any train_*.py script. Builds the manifest from the
CSV logs and image folders, extracts the flattened greyscale array for
every image (raw and delta), and caches the result to .npz files in
output_dir.

This is the slow, image-processing part of the pipeline. Every train_*.py
script just loads these cached arrays, so that cost is paid once, not
once per model.

Usage:
    python build_features.py
"""
from __future__ import annotations

import glob
import os
import re

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from common import CONFIG, features_path
from cv_utils import crop_fractional_roi

FORCE_COLUMNS = ["Fx", "Fy", "Fz"]

FOLDER_RE = re.compile(
    r"Cycle_r(?P<row>\d+)c(?P<col>\d+)_(?P<cycle>\d+)_(?P<phase>\w+)_(?P<depth>[\d.]+)mm"
)


def parse_folder_name(folder_name: str) -> dict:
    match = FOLDER_RE.search(folder_name)
    if match is None:
        return None
    return {
        "row": int(match.group("row")),
        "col": int(match.group("col")),
        "cycle": int(match.group("cycle")),
        "phase": match.group("phase"),
        "depth": float(match.group("depth")),
    }


def build_manifest(root_dir: str) -> pd.DataFrame:
    csv_paths = sorted(glob.glob(os.path.join(root_dir, "*.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV logs found in {root_dir}")

    rows = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if "image_file" not in df.columns:
            continue

        for image_folder, group in df.groupby("image_file"):
            folder_meta = parse_folder_name(os.path.basename(image_folder))
            if folder_meta is None:
                continue

            averaged = group[FORCE_COLUMNS].mean()
            folder_path = os.path.join(root_dir, image_folder)
            image_files = sorted(
                glob.glob(os.path.join(folder_path, "*.jpg"))
                + glob.glob(os.path.join(folder_path, "*.jpeg"))
                + glob.glob(os.path.join(folder_path, "*.png"))
            )
            for image_path in image_files:
                rows.append({
                    "image_path": image_path,
                    "row": folder_meta["row"],
                    "col": folder_meta["col"],
                    "cycle": folder_meta["cycle"],
                    "phase": folder_meta["phase"],
                    "depth": folder_meta["depth"],
                    "Fx": averaged["Fx"],
                    "Fy": averaged["Fy"],
                    "Fz": averaged["Fz"],
                })

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError("Manifest ended up empty, check root_dir and folder naming.")
    return manifest


def extract_flat_greyscale(image_path: str, active_roi_frac: tuple, fringe_roi_frac: tuple,
                            flatten_size: tuple) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    active = crop_fractional_roi(image, active_roi_frac)
    fringe = crop_fractional_roi(active, fringe_roi_frac)
    grey = cv2.cvtColor(fringe, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(grey, flatten_size, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32).flatten() / 255.0


def main():
    cfg = CONFIG
    os.makedirs(cfg["output_dir"], exist_ok=True)

    print("Building manifest from", cfg["root_dir"])
    manifest = build_manifest(cfg["root_dir"])
    print(f"Manifest has {len(manifest)} images across "
          f"{manifest[['row', 'col']].drop_duplicates().shape[0]} grid points.")

    print(f"Extracting greyscale arrays for {len(manifest)} images, this is the slow part...")
    features = np.stack([
        extract_flat_greyscale(p, cfg["active_roi_frac"], cfg["fringe_roi_frac"], cfg["flatten_size"])
        for p in manifest["image_path"]
    ])

    y = manifest[FORCE_COLUMNS].to_numpy()
    groups = (manifest["row"].astype(str) + "_" + manifest["col"].astype(str)).to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=cfg["val_fraction"], random_state=cfg["seed"])
    train_idx, val_idx = next(splitter.split(np.zeros(len(groups)), groups=groups))
    print(f"Group-aware split: {len(train_idx)} train, {len(val_idx)} val.")

    np.savez(
        features_path("raw"),
        X_train=features[train_idx], X_val=features[val_idx],
        y_train=y[train_idx], y_val=y[val_idx],
    )
    print(f"Saved {features_path('raw')}")

    print("Building delta features (each grid point's shallowest-depth frame subtracted out)...")
    baseline_lookup = {}
    for (row, col, cycle), group in manifest.groupby(["row", "col", "cycle"]):
        shallowest_idx = group["depth"].idxmin()
        baseline_lookup[(row, col, cycle)] = features[manifest.index.get_loc(shallowest_idx)]

    deltas = np.zeros_like(features)
    for i, (_, sample) in enumerate(manifest.iterrows()):
        key = (sample["row"], sample["col"], sample["cycle"])
        deltas[i] = features[i] - baseline_lookup[key]

    np.savez(
        features_path("delta"),
        X_train=deltas[train_idx], X_val=deltas[val_idx],
        y_train=y[train_idx], y_val=y[val_idx],
    )
    print(f"Saved {features_path('delta')}")
    print("\nDone. Every train_*.py script can now be run independently, in any order, "
          "including at the same time in separate terminals.")


if __name__ == "__main__":
    main()
