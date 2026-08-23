"""
Configuration for the optical tactile sensor force-regression pipeline.

Edit the paths and calibration values below before running train.py.
Nothing in this pipeline writes into CSV_PATH or IMAGE_ROOT; every
generated file (checkpoints, logs, plots, prediction CSVs) goes under
OUTPUT_DIR instead.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CSV_PATH = Path(r"D:\ERP\Data Analysis\pipeline_output\dataset.csv")
IMAGE_ROOT = Path(r"D:\ERP\Data Analysis\pipeline_output\crops")
OUTPUT_DIR = Path(r"D:\ERP\Data Analysis 2")

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
PLOT_DIR = OUTPUT_DIR / "plots"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
CALIBRATION_DIR = OUTPUT_DIR / "calibration"

for directory in (CHECKPOINT_DIR, LOG_DIR, PLOT_DIR, PREDICTIONS_DIR, CALIBRATION_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# CSV column names, matching the unified dataset.csv layout
# ---------------------------------------------------------------------------
COL_FRINGE_PATH = "fringe_delta_path"
COL_INDENT_PATH = "indent_delta_path"
COL_DEPTH = "depth_mm"
COL_X = "x_mm"
COL_Y = "y_mm"
COL_FX = "Fx"
COL_FY = "Fy"
COL_FZ = "Fz"
COL_ROW = "row"
COL_COL = "col"
COL_CYCLE = "cycle"
COL_PHASE = "phase"       # NOTE: holds direction ("up"/"down"), not a frame number
COL_IMG_IDX = "img_idx"   # the actual per-shot frame index (0-24)

# ---------------------------------------------------------------------------
# Legacy OpenCV pixel -> physical calibration
# ---------------------------------------------------------------------------
# NOT used by the main pipeline (dataset.py / train.py) any more. x_mm/y_mm
# turned out to be the fixed test-grid position (row, col), not something
# recoverable from a single re-centred crop, so contact location is now a
# learned CNN prediction (model.LocalisationCNN) instead of an OpenCV
# heuristic. These settings only exist for fit_pixel_calibration.py and
# diagnose_localisation.py, kept as a baseline/comparison reference.
CALIBRATION_PATH = CALIBRATION_DIR / "pixel_to_mm_calibration.json"

PIXEL_TO_MM_CALIBRATION = {
    "origin_u": 0.0,
    "origin_v": 0.0,
    "scale_x": 1.0,
    "scale_y": 1.0,
}

# ---------------------------------------------------------------------------
# Image parameters
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224  # resized square input fed to the CNN backbone
GRAYSCALE = True  # fringe/indent delta images are single-channel

# ---------------------------------------------------------------------------
# Raw photos: fixed ROI crop + delta preprocessing for predict_visual.py
# ---------------------------------------------------------------------------
# Folder of full, uncropped raw photos (before ROI cropping and delta
# computation), e.g. "D:\ERP\Pictures for Dataset\images". Only used by
# preview_rois.py and predict_visual.py; the main train.py/dataset.py
# pipeline never touches raw photos, only the pre-processed crops in
# IMAGE_ROOT.
RAW_IMAGES_ROOT = Path(r"D:\ERP\Pictures for Dataset\images")

# Confirmed empirically via search_roi_offset.py against a real shot
# (correlation 1.000 at the exact integer pixel offset). Full float
# precision here, combined with cv_utils.crop_fractional_roi now using
# round() instead of int(), avoids the 1px truncation error that degraded
# indent's delta match earlier.
FRINGE_ROI_FRAC = (0.19791666666666666, 0.05555555555555555, 0.9375, 0.4444444444444444)
INDENT_ROI_FRAC = (0.3645833333333333, 0.5185185185185185, 0.7291666666666666, 0.9074074074074074)

# Depth (mm) treated as "no contact" when auto-resolving a baseline
# (zero-indentation) raw photo for delta computation.
BASELINE_DEPTH_MM = 0.0

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 100
VAL_SPLIT = 0.15
TEST_SPLIT = 0.10
RANDOM_SEED = 42
DEPTH_LOSS_WEIGHT = 1.0
LOCATION_LOSS_WEIGHT = 1.0
FORCE_LOSS_WEIGHT = 1.0
HUBER_DELTA = 1.0
EARLY_STOPPING_PATIENCE = 15
