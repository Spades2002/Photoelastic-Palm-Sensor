r"""
Preview the fringe/indent ROI boxes against a real raw photo, so you can
tune config.FRINGE_ROI_FRAC / config.INDENT_ROI_FRAC before predict_visual.py
relies on them.

The current values in config.py are placeholders (left half / right half).
Run this, look at the saved preview image, adjust the fractions in
config.py, and re-run until the green box sits on the fringe region and
the blue box sits on the indent region.

Read-only: only reads one raw photo. Writes the preview PNG to
config.PREDICTIONS_DIR.

Run with:
    python preview_rois.py
    python preview_rois.py --image "D:\path\to\a\specific\raw\photo.jpg"
"""
from __future__ import annotations

import argparse
import logging

import config
from cv_utils import visualise_rois

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def find_sample_image() -> str:
    """Find any one raw photo under config.RAW_IMAGES_ROOT to preview against."""
    root = config.RAW_IMAGES_ROOT
    if not root.exists():
        raise SystemExit(
            f"config.RAW_IMAGES_ROOT does not exist: {root}\n"
            "Set it to your raw photos folder in config.py first."
        )
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            return str(path)
    raise SystemExit(f"No image files found under {root}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", default=None, help="Specific raw photo to preview against. "
                         "Default: the first image found under config.RAW_IMAGES_ROOT.")
    args = parser.parse_args()

    image_path = args.image or find_sample_image()
    logger.info("Previewing ROIs against: %s", image_path)
    logger.info("Current FRINGE_ROI_FRAC (green): %s", config.FRINGE_ROI_FRAC)
    logger.info("Current INDENT_ROI_FRAC (blue):  %s", config.INDENT_ROI_FRAC)

    out_path = config.PREDICTIONS_DIR / "roi_preview.png"
    visualise_rois(
        image_path,
        green_roi_frac=config.FRINGE_ROI_FRAC,
        blue_roi_frac=config.INDENT_ROI_FRAC,
        save_path=str(out_path),
    )
    logger.info("Saved preview to %s", out_path)
    logger.info("Open it, check the boxes, adjust FRINGE_ROI_FRAC/INDENT_ROI_FRAC in config.py, and re-run.")


if __name__ == "__main__":
    main()
