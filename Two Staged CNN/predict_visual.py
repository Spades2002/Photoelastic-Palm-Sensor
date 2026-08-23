r"""
Given one raw photo, crops the fringe/indent ROIs (config.FRINGE_ROI_FRAC /
config.INDENT_ROI_FRAC), computes a background-subtracted delta against a
baseline (no-contact) photo, runs the trained model, and saves an annotated
copy of the raw photo with the two ROI boxes drawn and predicted (and true,
if resolvable) depth/location/force printed as text.

Two things this deliberately does NOT do, and why:

1. No pixel crosshair for predicted (x_mm, y_mm). Earlier diagnosis on this
   dataset showed a classical OpenCV centroid on the pre-processed crops
   correlates poorly with true position (R^2 near zero) even though a
   learned CNN succeeds (R^2 ~0.998); we don't have evidence a naive
   pixel-mapping would be trustworthy here, so this script reports the
   number only, not a marker that implies false precision.
2. No guarantee the delta formula (cv_utils.compute_delta_crop, currently
   absolute difference) matches whatever your original pipeline used to
   generate fringe_delta_path/indent_delta_path. Since we don't have that
   pipeline's code, this script includes a built-in sanity check: if the
   raw photo you give it corresponds to a row already in dataset.csv, it
   loads the *real* pre-processed delta image for that row and reports how
   closely our from-scratch delta matches it. Trust the model's predictions
   more if that match is close; be skeptical if it isn't.

Usage:
    python predict_visual.py --image "D:\path\to\raw\photo.jpg"
    python predict_visual.py --image "D:\path\to\raw\photo.jpg" --baseline-image "D:\path\to\baseline.jpg"
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

import config
from cv_utils import compute_delta_crop, crop_fractional_roi
from model import TactileForceNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Matches the same "r{row}c{col}_{cycle}_{up|down}_{depth}mm" pattern seen
# in both the raw-photo folder names and the processed crop filenames, e.g.
# "Cycle_r10c8_5_up_0.000mm" or "r4c6_1_down_0.000mm_00_fringe_delta.png".
# Adjust this regex if your actual folder/filename convention differs.
SHOT_PATTERN = re.compile(r"r(\d+)c(\d+)_(\d+)_(up|down)_([\d.]+)mm")


def parse_shot_info(path_component: str) -> dict | None:
    """Extract row/col/cycle/direction/depth_mm from a folder or file name."""
    match = SHOT_PATTERN.search(path_component)
    if match is None:
        return None
    row, col, cycle, direction, depth = match.groups()
    return {
        "row": int(row), "col": int(col), "cycle": int(cycle),
        "direction": direction, "depth_mm": float(depth),
    }


def find_baseline_image(image_path: Path) -> Path | None:
    """Auto-resolve a no-contact baseline photo: same row/col/cycle, depth
    at config.BASELINE_DEPTH_MM, searched in the parent of image_path's
    folder (siblings of the current shot's folder). Prefers the same
    direction (up/down) if both exist."""
    shot = parse_shot_info(image_path.parent.name)
    if shot is None:
        logger.warning("Could not parse row/col/cycle from folder name %r; can't auto-resolve a baseline.",
                        image_path.parent.name)
        return None

    candidates = []
    for sibling in image_path.parent.parent.iterdir():
        if not sibling.is_dir():
            continue
        info = parse_shot_info(sibling.name)
        if info is None:
            continue
        if (info["row"], info["col"], info["cycle"]) != (shot["row"], shot["col"], shot["cycle"]):
            continue
        if abs(info["depth_mm"] - config.BASELINE_DEPTH_MM) > 1e-6:
            continue
        candidates.append((info["direction"] == shot["direction"], sibling))

    if not candidates:
        return None
    candidates.sort(key=lambda c: not c[0])  # same-direction match first
    baseline_folder = candidates[0][1]

    images = sorted(
        p for p in baseline_folder.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    )
    return images[0] if images else None


def find_point_csv_path(row: int, col: int, cycle: int) -> Path | None:
    """Locate the per-point CSV/XLSX log (e.g. Cycle_Indent5mm_r4c8_3.csv),
    which lives alongside the raw images folder, one level up from
    config.RAW_IMAGES_ROOT. Tries .csv first, then .xlsx."""
    search_root = config.RAW_IMAGES_ROOT.parent
    pattern = f"*r{row}c{col}_{cycle}.*"
    matches = sorted(search_root.glob(pattern))
    for ext in (".csv", ".xlsx"):
        for m in matches:
            if m.suffix.lower() == ext:
                return m
    return None


def find_true_values_from_point_csv(shot: dict) -> dict | None:
    """Fallback ground truth source when dataset.csv has no matching row.
    Reads the per-point log directly and averages Fx/Fy/Fz over the (few)
    timestamped rows at this exact indent_mm + phase, since there's no
    per-image (00-24) column to match a specific frame against, only a
    per-folder image_file reference. x_mm/y_mm are constant per point, so
    just take the first value."""
    point_csv = find_point_csv_path(shot["row"], shot["col"], shot["cycle"])
    if point_csv is None:
        return None

    if point_csv.suffix.lower() == ".xlsx":
        try:
            df = pd.read_excel(point_csv)
        except ImportError:
            logger.warning("Found %s but openpyxl isn't installed to read .xlsx; "
                            "pip install openpyxl, or export it to .csv.", point_csv)
            return None
    else:
        df = pd.read_csv(point_csv)

    required = {"indent_mm", "phase", "Fx", "Fy", "Fz", "x_mm", "y_mm"}
    if not required.issubset(df.columns):
        logger.warning("%s is missing expected columns %s", point_csv, required - set(df.columns))
        return None

    mask = np.isclose(df["indent_mm"], shot["depth_mm"]) & (df["phase"] == shot["direction"])
    matches = df[mask]
    if matches.empty:
        return None

    logger.info("True values sourced from %s (averaged over %d rows at this depth/phase)",
                point_csv, len(matches))
    return {
        "depth_mm": float(shot["depth_mm"]),
        "x_mm": float(matches["x_mm"].iloc[0]), "y_mm": float(matches["y_mm"].iloc[0]),
        "Fx": float(matches["Fx"].mean()), "Fy": float(matches["Fy"].mean()), "Fz": float(matches["Fz"].mean()),
        "fringe_delta_path": None, "indent_delta_path": None,
    }


def find_true_values(image_path: Path) -> dict | None:
    """If dataset.csv has a row matching this raw photo's row/col/cycle/depth
    (and phase, if the filename stem is numeric), return its ground truth.
    Falls back to the per-point CSV/XLSX log if dataset.csv has no match,
    since that fallback can't provide fringe_delta_path/indent_delta_path,
    so the delta sanity check only runs when dataset.csv itself matches."""
    shot = parse_shot_info(image_path.parent.name)
    if shot is None:
        return None

    if config.CSV_PATH.exists():
        df = pd.read_csv(config.CSV_PATH)
        required = {config.COL_ROW, config.COL_COL, config.COL_CYCLE, config.COL_DEPTH}
        if required.issubset(df.columns):
            mask = (
                (df[config.COL_ROW] == shot["row"])
                & (df[config.COL_COL] == shot["col"])
                & (df[config.COL_CYCLE] == shot["cycle"])
                & (np.isclose(df[config.COL_DEPTH], shot["depth_mm"]))
            )
            # config.COL_PHASE holds direction ("up"/"down"), not a frame
            # number, use it to disambiguate direction if both exist at
            # this depth.
            if config.COL_PHASE in df.columns:
                mask &= (df[config.COL_PHASE].astype(str).str.lower() == shot["direction"])
            # The actual per-shot frame index (0-24) lives in COL_IMG_IDX.
            if image_path.stem.isdigit() and config.COL_IMG_IDX in df.columns:
                mask &= (df[config.COL_IMG_IDX] == int(image_path.stem))

            matches = df[mask]
            if not matches.empty:
                row = matches.iloc[0]
                logger.info("True values sourced from dataset.csv")
                return {
                    "depth_mm": float(row[config.COL_DEPTH]),
                    "x_mm": float(row[config.COL_X]), "y_mm": float(row[config.COL_Y]),
                    "Fx": float(row[config.COL_FX]), "Fy": float(row[config.COL_FY]), "Fz": float(row[config.COL_FZ]),
                    "fringe_delta_path": row.get(config.COL_FRINGE_PATH),
                    "indent_delta_path": row.get(config.COL_INDENT_PATH),
                }

    logger.info("No matching row in dataset.csv; trying the per-point CSV/XLSX log instead.")
    return find_true_values_from_point_csv(shot)


def sanity_check_delta(computed_delta: np.ndarray, real_delta_path, image_root: Path, label: str) -> None:
    """If we know the real pre-processed delta file for this exact shot,
    compare it against what compute_delta_crop produced from scratch."""
    if real_delta_path is None or (isinstance(real_delta_path, float) and np.isnan(real_delta_path)):
        logger.info("%s delta sanity check skipped: no delta file path available "
                     "(true values came from the per-point CSV, not dataset.csv).", label)
        return
    candidate = Path(str(real_delta_path))
    if not candidate.is_absolute():
        candidate = image_root / candidate
    if not candidate.exists():
        candidate = image_root / Path(str(real_delta_path)).name
    if not candidate.exists():
        logger.info("Could not locate the real %s delta file for a sanity check.", label)
        return

    real = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
    if real is None:
        return
    resized = cv2.resize(computed_delta, (real.shape[1], real.shape[0]))
    mad = float(np.mean(np.abs(resized.astype(np.int16) - real.astype(np.int16))))
    logger.info(
        "%s delta sanity check: mean abs difference vs real pipeline output = %.1f "
        "(0-255 scale; low = our delta formula matches, high = it likely doesn't)",
        label, mad,
    )


def preprocess_crop(gray_delta: np.ndarray) -> torch.Tensor:
    resized = cv2.resize(gray_delta, (config.IMAGE_SIZE, config.IMAGE_SIZE))
    return torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, help="Path to the raw photo.")
    parser.add_argument("--baseline-image", default=None,
                         help="No-contact baseline raw photo. Auto-resolved from folder naming if omitted.")
    parser.add_argument("--checkpoint", default=str(config.CHECKPOINT_DIR / "best_model.pt"))
    parser.add_argument("--out", default=None, help="Output PNG path. Default: predictions/visual_<name>.png")
    args = parser.parse_args()

    image_path = Path(args.image)
    raw = cv2.imread(str(image_path))
    if raw is None:
        raise SystemExit(f"Could not read image: {image_path}")

    baseline_path = Path(args.baseline_image) if args.baseline_image else find_baseline_image(image_path)
    if baseline_path is None:
        raise SystemExit(
            "Could not auto-resolve a baseline (no-contact) photo. "
            "Pass one explicitly with --baseline-image."
        )
    baseline_raw = cv2.imread(str(baseline_path))
    if baseline_raw is None:
        raise SystemExit(f"Could not read baseline image: {baseline_path}")
    logger.info("Using baseline: %s", baseline_path)

    fringe_current = cv2.cvtColor(crop_fractional_roi(raw, config.FRINGE_ROI_FRAC), cv2.COLOR_BGR2GRAY)
    fringe_baseline = cv2.cvtColor(crop_fractional_roi(baseline_raw, config.FRINGE_ROI_FRAC), cv2.COLOR_BGR2GRAY)
    indent_current = cv2.cvtColor(crop_fractional_roi(raw, config.INDENT_ROI_FRAC), cv2.COLOR_BGR2GRAY)
    indent_baseline = cv2.cvtColor(crop_fractional_roi(baseline_raw, config.INDENT_ROI_FRAC), cv2.COLOR_BGR2GRAY)

    fringe_delta = compute_delta_crop(fringe_current, fringe_baseline)
    indent_delta = compute_delta_crop(indent_current, indent_baseline)

    true_values = find_true_values(image_path)
    if true_values:
        sanity_check_delta(fringe_delta, true_values.get("fringe_delta_path"), config.IMAGE_ROOT, "Fringe")
        sanity_check_delta(indent_delta, true_values.get("indent_delta_path"), config.IMAGE_ROOT, "Indent")
    else:
        logger.info("No matching dataset.csv row found for this shot; skipping the delta sanity check "
                     "and true-value comparison.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TactileForceNet(pretrained_backbone=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    fringe_tensor = preprocess_crop(fringe_delta).to(device)
    indent_tensor = preprocess_crop(indent_delta).to(device)
    with torch.no_grad():
        outputs = model(fringe_tensor, indent_tensor)
    depth_pred = outputs["depth_pred"].item()
    xy_pred = outputs["xy_pred"].squeeze(0).cpu().numpy()
    force_pred = outputs["force_pred"].squeeze(0).cpu().numpy()

    annotated = raw.copy()
    h, w = raw.shape[:2]
    for roi_frac, colour in ((config.FRINGE_ROI_FRAC, (0, 255, 0)), (config.INDENT_ROI_FRAC, (255, 0, 0))):
        x1, y1, x2, y2 = roi_frac
        cv2.rectangle(annotated, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), colour, 3)

    lines = [
        f"Predicted: depth={depth_pred:.2f}mm  x={xy_pred[0]:.1f}mm  y={xy_pred[1]:.1f}mm  "
        f"Fz={force_pred[2]:.2f}N  |F|={force_pred[3]:.2f}N"
    ]
    if true_values:
        loc_err = float(np.hypot(xy_pred[0] - true_values["x_mm"], xy_pred[1] - true_values["y_mm"]))
        lines.append(
            f"True:      depth={true_values['depth_mm']:.2f}mm  x={true_values['x_mm']:.1f}mm  "
            f"y={true_values['y_mm']:.1f}mm  Fz={true_values['Fz']:.2f}N"
        )
        lines.append(f"Depth error: {depth_pred - true_values['depth_mm']:+.2f}mm | Location error: {loc_err:.2f}mm")
    else:
        lines.append("(No matching CSV row found, true values unavailable)")

    y0 = h - 20 - 30 * (len(lines) - 1)
    for i, line in enumerate(lines):
        y = y0 + i * 30
        cv2.putText(annotated, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(annotated, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    out_path = Path(args.out) if args.out else config.PREDICTIONS_DIR / f"visual_{image_path.stem}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated)
    logger.info("Saved annotated image to %s", out_path)
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
