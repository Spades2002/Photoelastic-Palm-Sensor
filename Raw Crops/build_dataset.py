"""
Walks the whole dataset, extracts fringe + indentation features for every
photo (up to --max-images-per-shot of them), and writes one row per photo to
dataset.csv. Each row carries:

  - identifying metadata: point_id, row, col, cycle, phase, depth_mm
  - ground-truth labels from the CSV: Fx_mean, Fy_mean, Fz_mean, x_mm, y_mm
  - hand-crafted OpenCV features from feature_extraction.py
  - (optionally) paths to saved fringe/indent crop images, for CNN training,
    both the raw crop and a baseline-subtracted "delta" crop (current frame
    minus the point's own 0mm frame, shifted to mid-grey so it can still be
    saved as a normal image). The delta crop isolates what actually changed
    under load rather than making the network infer it from two full images
    it never sees side by side; try training on it via train_resnet.py's
    --crop-source delta.

All 25-ish photos in a shot share the same force/location label: the
indenter is held static during that dwell, so it is the shot, not the
individual frame, that has one physical state. Treating each photo as a
separate training example is intentional data augmentation, not a labelling
mistake.

Usage:
    python build_dataset.py [--max-images-per-shot 25] [--save-crops] [--phases down,up]

If your 'up' phase turns out to be a single retraction checkpoint rather
than a step-wise unloading curve (check with dataset_index.py: does the
'up' phase have anywhere near as many unique depths as 'down'?), you may
want --phases down to skip processing photos that add little variety.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import config
import dataset_index
import feature_extraction as fe

logger = logging.getLogger(__name__)


def _delta_crop(current_crop: np.ndarray, baseline_crop: np.ndarray) -> np.ndarray:
    """
    Signed difference (current - baseline), shifted by 128 so it's still a
    valid 0-255 image: mid-grey means no change, brighter means the pixel
    got lighter under load, darker means it got darker. Plain absolute
    difference would lose the sign; this keeps it, at the cost of halving
    the usable dynamic range either side of 128.
    """
    diff = current_crop.astype(np.int16) - baseline_crop.astype(np.int16)
    return np.clip(diff + 128, 0, 255).astype(np.uint8)


def _baseline_lookup(shots: pd.DataFrame) -> dict:
    """Map (row, col, cycle, phase) -> first image path of its 0mm baseline shot."""
    baselines = {}
    base = shots[shots["depth_mm"] <= config.BASELINE_DEPTH_MM]
    for _, r in base.iterrows():
        if r["n_images"] == 0:
            continue
        key = (r["row"], r["col"], r["cycle"], r["phase"])
        baselines[key] = r["images"][0]
    return baselines


def build(max_images_per_shot: int = 25, save_crops: bool = False, phases: tuple[str, ...] = ("down", "up"),
          crop_types: tuple[str, ...] = ("delta",)) -> pd.DataFrame:
    roi = fe.RoiConfig.load()
    result = dataset_index.build_index()
    shots = result.shots
    if shots.empty:
        raise SystemExit("No shots found. Check DATASET_ROOT / IMAGES_ROOT in config.py.")

    # Baselines are looked up from the full, unfiltered shot table, so a
    # down-only run can still baseline against the 0mm down frame even if
    # 'up' is excluded from the rows actually processed below.
    baselines = _baseline_lookup(shots)
    shots = shots[shots["phase"].isin(phases)].reset_index(drop=True)
    if shots.empty:
        raise SystemExit(f"No shots left after filtering to phases={phases}.")
    baseline_img_cache: dict[str, "object"] = {}

    if save_crops:
        config.CROPS_ROOT.mkdir(parents=True, exist_ok=True)

    rows = []
    n_shots = len(shots)
    for i, shot in shots.iterrows():
        key = (shot["row"], shot["col"], shot["cycle"], shot["phase"])
        baseline_path = baselines.get(key)
        baseline_img = None
        if baseline_path:
            if baseline_path not in baseline_img_cache:
                baseline_img_cache[baseline_path] = fe.load_image(baseline_path)
            baseline_img = baseline_img_cache[baseline_path]

        image_paths = shot["images"][:max_images_per_shot]
        for img_idx, img_path in enumerate(image_paths):
            img = fe.load_image(img_path)
            if img is None:
                logger.warning("Could not read %s, skipping", img_path)
                continue

            feats = fe.extract_all_features(img_path, roi, baseline_img=baseline_img)
            if feats is None:
                continue

            row = {
                "point_id": shot["point_id"],
                "row": shot["row"], "col": shot["col"], "cycle": shot["cycle"],
                "phase": shot["phase"], "depth_mm": shot["depth_mm"], "img_idx": img_idx,
                "x_mm": shot["x_mm"], "y_mm": shot["y_mm"],
                "Fx": shot["Fx_mean"], "Fy": shot["Fy_mean"], "Fz": shot["Fz_mean"],
                "n_force_samples": shot["n_force_samples"],
                "source_image": img_path,
                "has_baseline": baseline_img is not None,
            }
            row.update(feats)

            if save_crops:
                fringe_crop = fe.crop(img, roi.fringe_roi)
                indent_crop = fe.crop(img, roi.indent_roi)
                stem = f"{shot['shot_id']}_{img_idx:02d}"

                if "raw" in crop_types:
                    fringe_path = config.CROPS_ROOT / f"{stem}_fringe.png"
                    indent_path = config.CROPS_ROOT / f"{stem}_indent.png"
                    cv2.imwrite(str(fringe_path), fringe_crop)
                    cv2.imwrite(str(indent_path), indent_crop)
                    row["fringe_crop_path"] = str(fringe_path)
                    row["indent_crop_path"] = str(indent_path)

                if "delta" in crop_types and baseline_img is not None:
                    baseline_fringe_crop = fe.crop(baseline_img, roi.fringe_roi)
                    baseline_indent_crop = fe.crop(baseline_img, roi.indent_roi)
                    fringe_delta_path = config.CROPS_ROOT / f"{stem}_fringe_delta.png"
                    indent_delta_path = config.CROPS_ROOT / f"{stem}_indent_delta.png"
                    cv2.imwrite(str(fringe_delta_path), _delta_crop(fringe_crop, baseline_fringe_crop))
                    cv2.imwrite(str(indent_delta_path), _delta_crop(indent_crop, baseline_indent_crop))
                    row["fringe_delta_path"] = str(fringe_delta_path)
                    row["indent_delta_path"] = str(indent_delta_path)

            rows.append(row)

        if (i + 1) % 10 == 0 or (i + 1) == n_shots:
            logger.info("Processed %d/%d shots", i + 1, n_shots)

    df = pd.DataFrame(rows)
    config.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DATASET_CSV_PATH, index=False)
    logger.info("Wrote %d rows to %s", len(df), config.DATASET_CSV_PATH)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-images-per-shot", type=int, default=25)
    parser.add_argument("--save-crops", action="store_true", help="Also save fringe/indent crop images to disk (needed for CNN training)")
    parser.add_argument("--phases", type=str, default="down,up", help="Comma-separated phases to include, e.g. 'down' to skip the low-variety up phase")
    parser.add_argument("--crop-types", type=str, default="delta",
                         help="Comma-separated: 'delta' (default, what train_resnet.py actually uses now, roughly half the disk of saving both), 'raw', or 'raw,delta' for both")
    args = parser.parse_args()
    build(
        max_images_per_shot=args.max_images_per_shot,
        save_crops=args.save_crops,
        phases=tuple(p.strip() for p in args.phases.split(",")),
        crop_types=tuple(c.strip() for c in args.crop_types.split(",")),
    )
