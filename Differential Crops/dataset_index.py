"""
Builds a single, tidy table that links every image folder ("shot") to its
force readings and metadata, by cross-referencing the CSV files against the
image_file column and the images/ folder on disk.

A "shot" is one (point, cycle, phase, depth) combination, i.e. one image
folder such as Cycle_r4c6_1_down_0.000mm. It typically has ~5 rows in the
CSV (fast force-logger samples taken during the dwell) and ~25 photos on
disk (a burst captured during the same dwell). We treat all of those as
describing the same physical state: the indenter held at a fixed depth and
position.

Run this module directly to sanity-check your paths:
    python dataset_index.py
"""
from __future__ import annotations

import re
import logging
from pathlib import Path
from dataclasses import dataclass

import pandas as pd

import config

logger = logging.getLogger(__name__)

_CSV_RE = re.compile(config.CSV_NAME_PATTERN)
_IMG_RE = re.compile(config.IMAGE_FOLDER_PATTERN)


@dataclass
class IndexResult:
    readings: pd.DataFrame   # one row per CSV row (raw force samples)
    shots: pd.DataFrame      # one row per image folder (aggregated, with image lists)


def _parse_csv_filename(path: Path) -> dict | None:
    m = _CSV_RE.match(path.name)
    if not m:
        return None
    d = m.groupdict()
    return {"csv_row": int(d["row"]), "csv_col": int(d["col"]), "csv_cycle": int(d["cycle"])}


def _parse_image_folder_name(name: str) -> dict | None:
    m = _IMG_RE.search(name)
    if not m:
        return None
    d = m.groupdict()
    return {
        "row": int(d["row"]),
        "col": int(d["col"]),
        "cycle": int(d["cycle"]),
        "phase": d["phase"],
        "depth_mm": float(d["depth"]),
    }


def list_images_in_folder(folder: Path) -> list[str]:
    """Return sorted image file paths inside a shot folder, empty list if missing."""
    if not folder.is_dir():
        return []
    files = [p for p in folder.iterdir() if p.suffix.lower() in config.IMAGE_EXTENSIONS]
    return sorted(str(p) for p in files)


def load_all_readings(dataset_root: Path | None = None, images_root: Path | None = None) -> pd.DataFrame:
    """
    Read every CSV in dataset_root and return one combined dataframe, one row
    per original CSV row, with the image_file column resolved to an absolute
    folder path and parsed into (row, col, cycle, phase, depth_mm).
    """
    dataset_root = Path(dataset_root or config.DATASET_ROOT)
    images_root = Path(images_root or config.IMAGES_ROOT)

    csv_paths = sorted(dataset_root.glob("*.csv"))
    if not csv_paths:
        logger.warning("No CSV files found directly under %s", dataset_root)

    frames = []
    for csv_path in csv_paths:
        name_info = _parse_csv_filename(csv_path)
        if name_info is None:
            logger.warning("Skipping CSV with unexpected filename: %s", csv_path.name)
            continue

        df = pd.read_csv(csv_path)
        required = {"t_s", "indent_mm", "phase", "Fx", "Fy", "Fz", "x_mm", "y_mm", "image_file"}
        missing = required - set(df.columns)
        if missing:
            logger.warning("Skipping %s, missing columns: %s", csv_path.name, missing)
            continue

        parsed = df["image_file"].apply(lambda s: _parse_image_folder_name(Path(str(s)).name))
        bad = parsed.isna()
        if bad.any():
            logger.warning("%s: %d row(s) with unparseable image_file, dropping them", csv_path.name, bad.sum())
        df = df[~bad].copy()
        parsed = parsed[~bad]

        parsed_df = pd.DataFrame(list(parsed.values), index=df.index)
        # The image_file-derived phase is the canonical one used for grouping;
        # keep the CSV's own phase column too, under a different name, so we
        # can flag any disagreement without a duplicate-column clash.
        df = df.rename(columns={"phase": "phase_csv"})
        df = pd.concat([df, parsed_df], axis=1)

        phase_mismatch = df["phase_csv"] != df["phase"]
        if phase_mismatch.any():
            logger.warning(
                "%s: %d row(s) where the CSV phase column disagrees with the image_file phase",
                csv_path.name, phase_mismatch.sum(),
            )

        # Cross-check the CSV filename's point/cycle against what the image_file
        # column says. They should always agree; if not, something is misfiled.
        mismatched = (df["row"] != name_info["csv_row"]) | (df["col"] != name_info["csv_col"]) | (
            df["cycle"] != name_info["csv_cycle"]
        )
        if mismatched.any():
            logger.warning(
                "%s: %d row(s) whose image_file point/cycle disagrees with the filename",
                csv_path.name, mismatched.sum(),
            )

        df["csv_path"] = str(csv_path)
        df["image_folder_name"] = df["image_file"].apply(lambda s: Path(str(s)).name)
        df["image_folder_path"] = df["image_folder_name"].apply(lambda n: str(images_root / n))
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    readings = pd.concat(frames, ignore_index=True)
    return readings


def aggregate_shots(readings: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the raw per-sample readings into one row per shot (image folder):
    mean/std force over the CSV rows in that dwell, the fixed x_mm/y_mm
    position, and the list of actual image files found on disk.
    """
    if readings.empty:
        return pd.DataFrame()

    group_cols = ["row", "col", "cycle", "phase", "depth_mm"]
    agg = readings.groupby(group_cols).agg(
        Fx_mean=("Fx", "mean"), Fx_std=("Fx", "std"),
        Fy_mean=("Fy", "mean"), Fy_std=("Fy", "std"),
        Fz_mean=("Fz", "mean"), Fz_std=("Fz", "std"),
        x_mm=("x_mm", "first"),
        y_mm=("y_mm", "first"),
        indent_mm=("indent_mm", "first"),
        n_force_samples=("Fz", "size"),
        image_folder_path=("image_folder_path", "first"),
        csv_path=("csv_path", "first"),
    ).reset_index()

    agg["images"] = agg["image_folder_path"].apply(lambda p: list_images_in_folder(Path(p)))
    agg["n_images"] = agg["images"].apply(len)
    agg["point_id"] = agg.apply(lambda r: f"r{r['row']}c{r['col']}", axis=1)
    agg["shot_id"] = agg.apply(lambda r: f"{r['point_id']}_{r['cycle']}_{r['phase']}_{r['depth_mm']:.3f}mm", axis=1)

    missing = agg[agg["n_images"] == 0]
    if len(missing):
        logger.warning("%d shot(s) reference an image folder with no images found on disk", len(missing))

    return agg


def build_index(dataset_root: Path | None = None, images_root: Path | None = None) -> IndexResult:
    readings = load_all_readings(dataset_root, images_root)
    shots = aggregate_shots(readings)
    return IndexResult(readings=readings, shots=shots)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = build_index()
    print(f"Raw CSV rows loaded : {len(result.readings)}")
    print(f"Shots (image folders): {len(result.shots)}")
    if len(result.shots):
        print(result.shots.head(10).to_string())
        print(f"Shots with zero images on disk: {(result.shots['n_images'] == 0).sum()}")
