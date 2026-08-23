"""
Central configuration for the tactile sensor dataset pipeline.

Edit DATASET_ROOT and IMAGES_ROOT to match your machine before running
anything else. Every other script reads paths and constants from here,
so this is the only file you should need to touch to point the
pipeline at your data.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths - change these to match your machine
# ---------------------------------------------------------------------------
DATASET_ROOT = Path(r"E:\Extended Research Project\Pictures for Dataset")   # folder holding the CSVs
IMAGES_ROOT = DATASET_ROOT / "images"                  # folder holding the per-shot image folders

OUTPUT_ROOT = Path(r"D:\ERP\Data Analysis\pipeline_output")  # everything this pipeline writes goes here
ROI_CONFIG_PATH = OUTPUT_ROOT / "roi_config.json"
DATASET_CSV_PATH = OUTPUT_ROOT / "dataset.csv"
CROPS_ROOT = OUTPUT_ROOT / "crops"                     # saved fringe/indentation crops, used for CNN training
MODELS_ROOT = OUTPUT_ROOT / "models"

# ---------------------------------------------------------------------------
# Filename patterns
# ---------------------------------------------------------------------------
# e.g. "Cycle_Indent5mm_r8c8_1.csv"
CSV_NAME_PATTERN = r"Cycle_Indent5mm_r(?P<row>\d+)c(?P<col>\d+)_(?P<cycle>\d+)\.csv"

# e.g. the value "images/Cycle_r4c6_1_down_0.000mm" stored in the CSV's image_file column
IMAGE_FOLDER_PATTERN = (
    r"Cycle_r(?P<row>\d+)c(?P<col>\d+)_(?P<cycle>\d+)_(?P<phase>down|up)_(?P<depth>[\d.]+)mm"
)

# ---------------------------------------------------------------------------
# Region of interest defaults
# ---------------------------------------------------------------------------
# Fallback pixel boxes (x, y, w, h), only used until roi_calibrate.py produces
# a real roi_config.json for your camera setup. Update these once calibrated.
DEFAULT_FRINGE_ROI = {"x": 380, "y": 60, "w": 1420, "h": 420}   # green box: light fringe on top surface
DEFAULT_INDENT_ROI = {"x": 700, "y": 560, "w": 700, "h": 420}   # blue circle: mirror view of the dimple

# Depth (mm) at or below which a frame is treated as unloaded and used as the
# per-point baseline for background subtraction.
BASELINE_DEPTH_MM = 0.0

# Image file extensions to look for inside each shot folder
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")
