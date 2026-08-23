"""
PyTorch Dataset for the optical tactile sensor's dual-view delta images.

Each row in dataset.csv already carries the paths to the fringe delta and
indent delta crops, along with the physics targets, so no filenames need to
be reconstructed here.

Depth, contact location (x_mm, y_mm) and force are all treated purely as
training labels here, never as model inputs: at deployment none of them
are known in advance, that's exactly what the sensor is for. So this
Dataset just returns both raw images plus the three label sets; the model
(model.py) is responsible for predicting all of them from pixels alone.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import config

logger = logging.getLogger(__name__)


def _load_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


class TactileSensorDataset(Dataset):
    """Reads the master CSV and, for each row, loads the fringe delta image
    and the indent delta image.

    Returned sample dict keys:
        fringe_image : FloatTensor (1, H, W), normalised to [0, 1]
        indent_image : FloatTensor (1, H, W), normalised to [0, 1]
        depth_mm     : FloatTensor scalar, ground-truth indentation depth (label only)
        xy_mm        : FloatTensor (2,), ground-truth (x_mm, y_mm) contact location (label only)
        force        : FloatTensor (4,), [Fx, Fy, Fz, magnitude] (label only)
        meta         : dict of row, col, cycle, phase
    """

    def __init__(
        self,
        csv_path: Path | str = config.CSV_PATH,
        image_root: Path | str = config.IMAGE_ROOT,
        image_size: int = config.IMAGE_SIZE,
        indices: list | None = None,
    ):
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root)
        self.image_size = image_size

        self.frame = pd.read_csv(self.csv_path)
        required_cols = [
            config.COL_FRINGE_PATH,
            config.COL_INDENT_PATH,
            config.COL_DEPTH,
            config.COL_X,
            config.COL_Y,
            config.COL_FX,
            config.COL_FY,
            config.COL_FZ,
        ]
        missing = [c for c in required_cols if c not in self.frame.columns]
        if missing:
            raise ValueError(f"Dataset CSV is missing expected columns: {missing}")

        if indices is not None:
            self.frame = self.frame.iloc[indices].reset_index(drop=True)

        self._path_strategy_index = self._detect_path_strategy()

    def __len__(self) -> int:
        return len(self.frame)

    def _candidate_path(self, raw_path: str, strategy_index: int) -> Path:
        """Build one candidate resolution of a CSV-stored image path.

        Strategy 0: the path exactly as stored (works if it's a still-valid
        absolute path, or relative to IMAGE_ROOT).
        Strategy 1: relative to IMAGE_ROOT's parent (covers CSVs that already
        include the crops folder name).
        Strategy 2: just the bare filename, joined onto IMAGE_ROOT. This is
        the one that survives moving the whole dataset to a new drive or
        folder, since it ignores whatever directory (or drive letter) the
        CSV originally recorded and only trusts IMAGE_ROOT plus the filename.
        """
        p = Path(str(raw_path))
        if strategy_index == 0:
            return p if p.is_absolute() else self.image_root / p
        if strategy_index == 1:
            return self.image_root.parent / p
        if strategy_index == 2:
            return self.image_root / p.name
        raise ValueError(f"Unknown path resolution strategy: {strategy_index}")

    def _detect_path_strategy(self) -> int:
        """Probe the first row against each known path convention and lock in
        whichever one actually resolves to real files on disk, so every
        subsequent __getitem__ call is a single, fast, correct join instead of
        repeatedly trying multiple candidates."""
        if len(self.frame) == 0:
            return 0

        row = self.frame.iloc[0]
        fringe_raw = row[config.COL_FRINGE_PATH]
        indent_raw = row[config.COL_INDENT_PATH]

        for idx in range(3):
            fringe_candidate = self._candidate_path(fringe_raw, idx)
            indent_candidate = self._candidate_path(indent_raw, idx)
            if fringe_candidate.exists() and indent_candidate.exists():
                logger.info("Resolved image paths using strategy %d (e.g. %s)", idx, fringe_candidate)
                return idx

        tried = [str(self._candidate_path(fringe_raw, i)) for i in range(3)]
        raise FileNotFoundError(
            "Could not resolve image paths from the CSV against IMAGE_ROOT using any "
            f"known convention. Tried: {tried}. Check config.IMAGE_ROOT and the path "
            "format actually stored in the CSV."
        )

    def _resolve_path(self, raw_path: str) -> Path:
        return self._candidate_path(raw_path, self._path_strategy_index)

    def _load_and_prepare(self, path: Path) -> torch.Tensor:
        image = _load_grayscale(path)
        image = cv2.resize(image, (self.image_size, self.image_size))
        return torch.from_numpy(image).float().unsqueeze(0) / 255.0

    def __getitem__(self, idx: int) -> dict:
        row = self.frame.iloc[idx]

        fringe_path = self._resolve_path(row[config.COL_FRINGE_PATH])
        indent_path = self._resolve_path(row[config.COL_INDENT_PATH])

        fringe_tensor = self._load_and_prepare(fringe_path)
        indent_tensor = self._load_and_prepare(indent_path)

        depth_mm = torch.tensor(float(row[config.COL_DEPTH]), dtype=torch.float32)
        xy_mm = torch.tensor(
            [float(row[config.COL_X]), float(row[config.COL_Y])], dtype=torch.float32
        )

        fx = float(row[config.COL_FX])
        fy = float(row[config.COL_FY])
        fz = float(row[config.COL_FZ])
        magnitude = float(np.sqrt(fx**2 + fy**2 + fz**2))
        force = torch.tensor([fx, fy, fz, magnitude], dtype=torch.float32)

        meta = {
            "row": row.get(config.COL_ROW, -1),
            "col": row.get(config.COL_COL, -1),
            "cycle": row.get(config.COL_CYCLE, -1),
            "phase": row.get(config.COL_PHASE, ""),
        }

        return {
            "fringe_image": fringe_tensor,
            "indent_image": indent_tensor,
            "depth_mm": depth_mm,
            "xy_mm": xy_mm,
            "force": force,
            "meta": meta,
        }
