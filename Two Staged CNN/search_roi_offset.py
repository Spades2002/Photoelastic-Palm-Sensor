r"""
Automatically finds the fringe/indent ROI position within the raw photo,
instead of hand-tuning FRINGE_ROI_FRAC/INDENT_ROI_FRAC by eye.

Now that diagnose_delta_formula.py has told us the real delta files' exact
pixel size (e.g. 1420x420 for fringe), we know the correct ROI SHAPE. What
we don't know is its POSITION within the raw photo. This script fixes the
window to that known size and slides it across the whole raw photo (coarse
stride first, then a finer pass around the best coarse result), computing a
candidate delta at each position and scoring its correlation against the
real delta file. The position with the highest correlation is very likely
the real ROI's actual location.

This only works for a shot that IS in dataset.csv (needs a real delta file
to search against). Once you have good FRINGE_ROI_FRAC/INDENT_ROI_FRAC
values from this, they apply to every shot, not just this one, since the
ROI is a fixed rectangle on the sensor rig.

Usage:
    python search_roi_offset.py --image "D:\path\to\raw\photo.jpg"
    python search_roi_offset.py --image "...\17.jpg" --coarse-stride 15 --fine-stride 2
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

import config
from predict_visual import find_baseline_image, parse_shot_info
from diagnose_delta_formula import load_real_delta_paths, resolve_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def candidate_deltas(current_win: np.ndarray, baseline_win: np.ndarray) -> list:
    """Same handful of formulas as diagnose_delta_formula.py, kept small
    since this runs many times per search."""
    c = current_win.astype(np.int16)
    b = baseline_win.astype(np.int16)
    return [
        np.abs(c - b).clip(0, 255).astype(np.uint8),
        (c - b + 128).clip(0, 255).astype(np.uint8),
        (b - c + 128).clip(0, 255).astype(np.uint8),
    ]


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    m = np.corrcoef(a.flatten().astype(np.float64), b.flatten().astype(np.float64))
    val = m[0, 1]
    return float(val) if not np.isnan(val) else -1.0


def search(current_gray: np.ndarray, baseline_gray: np.ndarray, real: np.ndarray,
           win_w: int, win_h: int, x_range: range, y_range: range) -> tuple:
    """Slide a (win_w, win_h) window over the given x/y offset ranges,
    return (best_x, best_y, best_corr)."""
    best = (-1, -1, -1.0)
    for y in y_range:
        for x in x_range:
            current_win = current_gray[y:y + win_h, x:x + win_w]
            baseline_win = baseline_gray[y:y + win_h, x:x + win_w]
            if current_win.shape != (win_h, win_w) or baseline_win.shape != (win_h, win_w):
                continue
            for candidate in candidate_deltas(current_win, baseline_win):
                corr = correlation(candidate, real)
                if corr > best[2]:
                    best = (x, y, corr)
    return best


def search_one(label: str, current_gray: np.ndarray, baseline_gray: np.ndarray,
                real_path: Path, coarse_stride: int, fine_stride: int) -> None:
    real = cv2.imread(str(real_path), cv2.IMREAD_GRAYSCALE)
    if real is None:
        logger.warning("Could not read real delta file for %s: %s", label, real_path)
        return

    win_h, win_w = real.shape[:2]
    frame_h, frame_w = current_gray.shape[:2]
    if win_w > frame_w or win_h > frame_h:
        logger.warning("%s window (%dx%d) is larger than the raw photo (%dx%d); skipping.",
                        label, win_w, win_h, frame_w, frame_h)
        return

    logger.info("--- %s: searching for a %dx%d window in a %dx%d photo ---", label, win_w, win_h, frame_w, frame_h)

    coarse_x = range(0, frame_w - win_w + 1, coarse_stride)
    coarse_y = range(0, frame_h - win_h + 1, coarse_stride)
    logger.info("  Coarse pass: %d x %d positions, stride %d...", len(coarse_x), len(coarse_y), coarse_stride)
    bx, by, bcorr = search(current_gray, baseline_gray, real, win_w, win_h, coarse_x, coarse_y)
    logger.info("  Coarse best: x=%d y=%d correlation=%.3f", bx, by, bcorr)

    fine_x = range(max(0, bx - coarse_stride), min(frame_w - win_w, bx + coarse_stride) + 1, fine_stride)
    fine_y = range(max(0, by - coarse_stride), min(frame_h - win_h, by + coarse_stride) + 1, fine_stride)
    logger.info("  Fine pass: %d x %d positions, stride %d...", len(fine_x), len(fine_y), fine_stride)
    fx, fy, fcorr = search(current_gray, baseline_gray, real, win_w, win_h, fine_x, fine_y)
    logger.info("  Fine best: x=%d y=%d correlation=%.3f", fx, fy, fcorr)

    x1_frac, y1_frac = fx / frame_w, fy / frame_h
    x2_frac, y2_frac = (fx + win_w) / frame_w, (fy + win_h) / frame_h
    logger.info(
        "  >>> Suggested %s_ROI_FRAC = (%r, %r, %r, %r)  [correlation %.3f]",
        label.upper(), x1_frac, y1_frac, x2_frac, y2_frac, fcorr,
    )
    logger.info(
        "  (Full float precision above is deliberate: cv_utils.crop_fractional_roi rounds to the "
        "nearest pixel, but a 4-decimal-rounded fraction can still land 1px off, which matters more "
        "for sharp features like the indent view than smooth ones like the fringe view.)"
    )
    if fcorr < 0.3:
        logger.warning(
            "  Correlation is still low even at the best position found. Either the delta formula "
            "candidates tried here aren't right either, or the search stride missed the true "
            "position, try a smaller --coarse-stride, or check the baseline image is correct."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, help="Path to a raw photo that IS in dataset.csv.")
    parser.add_argument("--baseline-image", default=None)
    parser.add_argument("--coarse-stride", type=int, default=20)
    parser.add_argument("--fine-stride", type=int, default=2)
    args = parser.parse_args()

    image_path = Path(args.image)
    raw = cv2.imread(str(image_path))
    if raw is None:
        raise SystemExit(f"Could not read image: {image_path}")
    logger.info("Raw photo size: %dx%d", raw.shape[1], raw.shape[0])

    baseline_path = Path(args.baseline_image) if args.baseline_image else find_baseline_image(image_path)
    if baseline_path is None:
        raise SystemExit("Could not auto-resolve a baseline photo. Pass one with --baseline-image.")
    baseline_raw = cv2.imread(str(baseline_path))
    if baseline_raw is None:
        raise SystemExit(f"Could not read baseline image: {baseline_path}")
    logger.info("Using baseline: %s", baseline_path)

    current_gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    baseline_gray = cv2.cvtColor(baseline_raw, cv2.COLOR_BGR2GRAY)

    fringe_real_ref, indent_real_ref = load_real_delta_paths(image_path)
    fringe_real_path = resolve_path(fringe_real_ref, config.IMAGE_ROOT)
    indent_real_path = resolve_path(indent_real_ref, config.IMAGE_ROOT)
    if fringe_real_path is None or indent_real_path is None:
        raise SystemExit(
            "This shot isn't in dataset.csv (or its delta files can't be found), "
            "so there's nothing to search against. Use a shot that IS in dataset.csv."
        )

    search_one("fringe", current_gray, baseline_gray, fringe_real_path, args.coarse_stride, args.fine_stride)
    search_one("indent", current_gray, baseline_gray, indent_real_path, args.coarse_stride, args.fine_stride)

    logger.info("Paste the suggested FRINGE_ROI_FRAC / INDENT_ROI_FRAC values into config.py, "
                "then run preview_rois.py to visually confirm before relying on them.")


if __name__ == "__main__":
    main()
