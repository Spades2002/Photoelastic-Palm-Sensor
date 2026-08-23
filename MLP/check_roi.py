"""
Visual check for active_roi_frac / fringe_roi_frac before running
build_features.py on real data.

Draws both boxes on a sample frame:
    blue  = active_roi_frac  (dead-space trim, relative to the raw frame)
    green = fringe_roi_frac  (force view, relative to the ACTIVE-cropped
            image, converted here into raw-frame coordinates so both boxes
            can be drawn on the same picture)

Also saves what the model actually receives after crop, greyscale
conversion, and resize to flatten_size, upscaled so it's viewable, since a
box that looks fine on the full frame can still end up as a poor,
low-detail patch once shrunk down.

Usage:
    python check_roi.py "D:\\ERP\\Pictures for Dataset\\images\\Cycle_r10c8_5_down_5.000mm\\00.jpg"

Reads ACTIVE_ROI_FRAC / FRINGE_ROI_FRAC / FLATTEN_SIZE straight from
common.py's CONFIG, so this always matches what build_features.py will
actually use, edit them there, not here.
"""
from __future__ import annotations

import sys

import cv2

from common import CONFIG
from cv_utils import crop_fractional_roi, visualise_rois

ACTIVE_ROI_FRAC = CONFIG["active_roi_frac"]
FRINGE_ROI_FRAC = CONFIG["fringe_roi_frac"]
FLATTEN_SIZE = CONFIG["flatten_size"]


def combined_fringe_frac(active_frac: tuple, fringe_frac: tuple) -> tuple:
    """fringe_roi_frac is relative to the active-cropped image. Convert it
    into a fraction of the RAW frame, so both boxes can be drawn on the
    same unmodified picture."""
    ax1, ay1, ax2, ay2 = active_frac
    fx1, fy1, fx2, fy2 = fringe_frac
    active_w, active_h = (ax2 - ax1), (ay2 - ay1)
    return (
        ax1 + fx1 * active_w,
        ay1 + fy1 * active_h,
        ax1 + fx2 * active_w,
        ay1 + fy2 * active_h,
    )


def main(image_path: str):
    combined_frac = combined_fringe_frac(ACTIVE_ROI_FRAC, FRINGE_ROI_FRAC)

    boxes_path = "roi_boxes_preview.jpg"
    visualise_rois(
        image_path=image_path,
        green_roi_frac=combined_frac,     # fringe view, in raw-frame coordinates
        blue_roi_frac=ACTIVE_ROI_FRAC,     # active region
        save_path=boxes_path,
    )
    print(f"Boxes drawn on the raw frame: {boxes_path}")
    print(f"  blue  = active_roi_frac  {ACTIVE_ROI_FRAC}")
    print(f"  green = fringe_roi_frac  {FRINGE_ROI_FRAC} "
          f"(shown here as {tuple(round(v, 4) for v in combined_frac)} of the raw frame)")

    # What the model actually receives: crop, greyscale, resize.
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    active = crop_fractional_roi(image, ACTIVE_ROI_FRAC)
    fringe = crop_fractional_roi(active, FRINGE_ROI_FRAC)
    grey = cv2.cvtColor(fringe, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(grey, FLATTEN_SIZE, interpolation=cv2.INTER_AREA)

    # Upscale purely for viewing, nearest-neighbour so you see the actual
    # pixels the model uses rather than a smoothed reinterpretation.
    upscale_factor = max(1, 300 // max(FLATTEN_SIZE))
    viewable = cv2.resize(
        resized, (FLATTEN_SIZE[0] * upscale_factor, FLATTEN_SIZE[1] * upscale_factor),
        interpolation=cv2.INTER_NEAREST,
    )
    model_input_path = "roi_model_input_preview.jpg"
    cv2.imwrite(model_input_path, viewable)
    print(f"What the model actually sees ({FLATTEN_SIZE[0]}x{FLATTEN_SIZE[1]}, "
          f"upscaled {upscale_factor}x for viewing): {model_input_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
