"""
Group-aware train/val/test split.

Splits at the level of physical "points" (by default, one row/col/cycle
combination, i.e. one full indentation cycle at one grid location) rather
than individual CSV rows, so every frame belonging to the same point (all
its depths, phases, and image indices) lands entirely in one split. A plain
random row-level split lets near-duplicate frames from the same shot, or
from a continuous depth sweep, leak across train and test, letting the
model partially "recognise" test examples it already saw a near-twin of
during training.

Uses sklearn's GroupShuffleSplit, applied twice: once to carve out the test
set, then again on the remainder to carve out the validation set. Every
script that needs "the held-out test set" should call get_group_split_indices
here rather than rolling its own split, so they all agree on exactly which
rows were held out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import config


def make_group_ids(frame: pd.DataFrame, group_cols: list) -> np.ndarray:
    """Combine group_cols into one string ID per row, e.g.
    row=9, col=8, cycle=2 -> "9_8_2". Rows sharing all group_cols values
    share a group ID and are guaranteed to land in the same split."""
    return frame[group_cols].astype(str).agg("_".join, axis=1).values


def get_group_split_indices(
    frame: pd.DataFrame,
    group_cols: list | None = None,
    val_frac: float | None = None,
    test_frac: float | None = None,
    seed: int | None = None,
) -> tuple:
    """Returns (train_idx, val_idx, test_idx): positional indices into
    frame (0-based, usable with frame.iloc[...] or torch.utils.data.Subset),
    split so that no group_cols combination appears in more than one split.

    Defaults come from config: GROUP_SPLIT_COLS, VAL_SPLIT, TEST_SPLIT,
    RANDOM_SEED, so every caller that doesn't override anything gets the
    exact same split.
    """
    group_cols = group_cols or config.GROUP_SPLIT_COLS
    val_frac = config.VAL_SPLIT if val_frac is None else val_frac
    test_frac = config.TEST_SPLIT if test_frac is None else test_frac
    seed = config.RANDOM_SEED if seed is None else seed

    groups = make_group_ids(frame, group_cols)
    n = len(frame)
    all_idx = np.arange(n)

    # First carve off the test set.
    test_splitter = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    train_val_idx, test_idx = next(test_splitter.split(all_idx, groups=groups))

    # Then split what's left into train/val, using the equivalent fraction
    # of the REMAINING data (so the final val fraction of the whole dataset
    # still matches config.VAL_SPLIT, not val_frac of the already-reduced
    # train_val subset).
    remaining_frac = 1.0 - test_frac
    val_frac_of_remaining = val_frac / remaining_frac
    val_splitter = GroupShuffleSplit(n_splits=1, test_size=val_frac_of_remaining, random_state=seed)
    train_sub_idx, val_sub_idx = next(val_splitter.split(train_val_idx, groups=groups[train_val_idx]))

    train_idx = train_val_idx[train_sub_idx]
    val_idx = train_val_idx[val_sub_idx]

    _assert_no_group_overlap(groups, train_idx, val_idx, test_idx)
    return train_idx, val_idx, test_idx


def _assert_no_group_overlap(groups: np.ndarray, train_idx, val_idx, test_idx) -> None:
    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])
    test_groups = set(groups[test_idx])
    overlap = (train_groups & val_groups) | (train_groups & test_groups) | (val_groups & test_groups)
    if overlap:
        raise RuntimeError(
            f"Group split produced {len(overlap)} group(s) present in more than one split, "
            "this should be impossible with GroupShuffleSplit; something is wrong."
        )
