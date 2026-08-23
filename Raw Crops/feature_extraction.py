"""
OpenCV-based feature extraction for the two regions you annotated:

  * fringe_roi  (green box)  - the flat top surface where light fringes
                                appear under load. Used as the force cue.
  * indent_roi  (blue circle) - the mirror view showing the dimple where
                                 the indenter presses in. Used as the
                                 location cue.

For each region we compute:
  1. Plain appearance features (colour statistics, edge density) from the
     current frame alone.
  2. Difference features against a per-point baseline frame (the 0.000mm,
     unloaded shot for the same point/cycle/phase). This is the more useful
     signal: it isolates exactly what changed when the indenter loaded the
     sensor, rather than fixed lighting/reflection clutter that is present
     in every frame regardless of load.

For the indentation region specifically, the dimple is located as the
largest dark blob in the difference image (falls back to the largest dark
blob in the raw crop if no baseline is available), giving a pixel centroid
that downstream code can compare against the CSV's ground-truth x_mm/y_mm.
"""
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, asdict

import cv2
import numpy as np

import config


@dataclass
class RoiConfig:
    fringe_roi: dict
    indent_roi: dict

    @classmethod
    def load(cls, path: Path | None = None) -> "RoiConfig":
        path = Path(path or config.ROI_CONFIG_PATH)
        if path.exists():
            with open(path) as f:
                d = json.load(f)
            return cls(fringe_roi=d["fringe_roi"], indent_roi=d["indent_roi"])
        return cls(fringe_roi=config.DEFAULT_FRINGE_ROI, indent_roi=config.DEFAULT_INDENT_ROI)


def load_image(path: str | Path) -> np.ndarray | None:
    img = cv2.imread(str(path))
    return img


def crop(img: np.ndarray, roi: dict) -> np.ndarray:
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    H, W = img.shape[:2]
    x2, y2 = min(x + w, W), min(y + h, H)
    x, y = max(x, 0), max(y, 0)
    return img[y:y2, x:x2]


def _safe_mean_std(arr: np.ndarray) -> tuple[float, float]:
    return float(np.mean(arr)), float(np.std(arr))


def extract_fringe_features(fringe_crop: np.ndarray, baseline_crop: np.ndarray | None = None) -> dict:
    """Colour/texture features from the fringe (force cue) region."""
    hsv = cv2.cvtColor(fringe_crop, cv2.COLOR_BGR2HSV)
    b_mean, b_std = _safe_mean_std(fringe_crop[:, :, 0])
    g_mean, g_std = _safe_mean_std(fringe_crop[:, :, 1])
    r_mean, r_std = _safe_mean_std(fringe_crop[:, :, 2])
    h_mean, h_std = _safe_mean_std(hsv[:, :, 0])
    s_mean, s_std = _safe_mean_std(hsv[:, :, 1])
    v_mean, v_std = _safe_mean_std(hsv[:, :, 2])

    gray = cv2.cvtColor(fringe_crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.mean(edges > 0))

    # Pixels that look like coloured fringe rather than the plain pale
    # surface: reasonably saturated, not near-black or near-white.
    fringe_mask = (hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 30) & (hsv[:, :, 2] < 240)
    fringe_pixel_ratio = float(np.mean(fringe_mask))

    feats = {
        "fringe_b_mean": b_mean, "fringe_g_mean": g_mean, "fringe_r_mean": r_mean,
        "fringe_h_mean": h_mean, "fringe_s_mean": s_mean, "fringe_v_mean": v_mean,
        "fringe_s_std": s_std, "fringe_v_std": v_std,
        "fringe_edge_density": edge_density,
        "fringe_pixel_ratio": fringe_pixel_ratio,
    }

    if baseline_crop is not None and baseline_crop.shape == fringe_crop.shape:
        diff = cv2.absdiff(fringe_crop, baseline_crop)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        feats["fringe_diff_mean"] = float(np.mean(diff_gray))
        feats["fringe_diff_max"] = float(np.max(diff_gray))
        feats["fringe_diff_area_ratio"] = float(np.mean(diff_gray > 25))
    else:
        feats["fringe_diff_mean"] = np.nan
        feats["fringe_diff_max"] = np.nan
        feats["fringe_diff_area_ratio"] = np.nan

    return feats


def _largest_dark_blob(mask: np.ndarray) -> dict | None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 4:
        return None
    M = cv2.moments(c)
    cx = M["m10"] / M["m00"] if M["m00"] else np.nan
    cy = M["m01"] / M["m00"] if M["m00"] else np.nan
    (_, _), radius = cv2.minEnclosingCircle(c)
    return {"cx": float(cx), "cy": float(cy), "area": float(area), "radius": float(radius)}


def extract_indentation_features(indent_crop: np.ndarray, baseline_crop: np.ndarray | None = None) -> dict:
    """Locate the dimple in the mirror view (location cue)."""
    gray = cv2.cvtColor(indent_crop, cv2.COLOR_BGR2GRAY)
    mean_intensity, std_intensity = _safe_mean_std(gray)

    feats = {
        "indent_mean_intensity": mean_intensity,
        "indent_std_intensity": std_intensity,
    }

    # Raw-frame blob: darker-than-average region, works even without a baseline
    # but is more easily confused by fixed reflections/shadows.
    thresh_val = max(mean_intensity - std_intensity, 0)
    raw_mask = gray < thresh_val
    raw_blob = _largest_dark_blob(raw_mask)

    diff_blob = None
    if baseline_crop is not None and baseline_crop.shape == indent_crop.shape:
        base_gray = cv2.cvtColor(baseline_crop, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, base_gray)
        feats["indent_diff_mean"] = float(np.mean(diff))
        feats["indent_diff_max"] = float(np.max(diff))
        _, diff_thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        diff_thresh = cv2.morphologyEx(diff_thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        diff_blob = _largest_dark_blob(diff_thresh > 0)
    else:
        feats["indent_diff_mean"] = np.nan
        feats["indent_diff_max"] = np.nan

    # Prefer the diff-based blob (isolates what changed under load); fall
    # back to the raw-frame blob if there is no baseline available.
    blob = diff_blob if diff_blob is not None else raw_blob
    source = "diff" if diff_blob is not None else ("raw" if raw_blob is not None else "none")

    feats["indent_blob_source"] = source
    feats["indent_px_x"] = blob["cx"] if blob else np.nan
    feats["indent_px_y"] = blob["cy"] if blob else np.nan
    feats["indent_px_radius"] = blob["radius"] if blob else np.nan
    feats["indent_px_area"] = blob["area"] if blob else np.nan

    return feats


def _deviation_from_delta_crop(delta_crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(delta_crop, cv2.COLOR_BGR2GRAY) if delta_crop.ndim == 3 else delta_crop
    return np.abs(gray.astype(np.int16) - 128).astype(np.uint8)


def _weighted_centroid_from_deviation(deviation: np.ndarray, min_area: int = 20) -> dict | None:
    """
    deviation: single-channel array, larger values = more change from
    baseline. Segments with Otsu's threshold (auto-picked per image from
    its own histogram, rather than one fixed cutoff across every point and
    depth), then computes both an intensity-weighted centroid (pixels
    closer to the peak count more) and a plain unweighted one (every pixel
    in the blob counts equally, like a simple binary-mask moment) within
    the largest connected region, so the two can be compared directly:
    intensity weighting is vulnerable to lighting asymmetry pulling the
    centroid toward whichever side is brighter, the unweighted version
    isn't.

    Returns None if nothing clears the threshold, deliberately, rather
    than falling back to a less reliable proxy: a clean "no detection"
    signal is more useful downstream than a wrong number that looks valid.
    """
    if deviation.dtype != np.uint8:
        deviation = np.clip(deviation, 0, 255).astype(np.uint8)

    thresh_val, mask = cv2.threshold(deviation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < min_area:
        return None

    blob_mask = np.zeros_like(mask)
    cv2.drawContours(blob_mask, [largest], -1, 255, -1)

    # unweighted: plain centroid of the blob mask, every pixel inside counts equally
    M = cv2.moments(blob_mask, binaryImage=True)
    cx_unweighted = float(M["m10"] / M["m00"]) if M["m00"] else None
    cy_unweighted = float(M["m01"] / M["m00"]) if M["m00"] else None

    # weighted: pixels closer to the deviation peak count more
    weights = deviation.astype(np.float64) * (blob_mask > 0)
    total_weight = weights.sum()
    if total_weight <= 0:
        return None
    ys, xs = np.indices(deviation.shape)
    cx = float((xs * weights).sum() / total_weight)
    cy = float((ys * weights).sum() / total_weight)
    # rough 0-1ish quality signal: how strong the change is, on average,
    # within the detected blob. Low values mean a faint, marginal detection.
    confidence = float(weights.sum() / 255.0 / max(area, 1))

    return {
        "cx": cx, "cy": cy,
        "cx_unweighted": cx_unweighted, "cy_unweighted": cy_unweighted,
        "area": float(area), "otsu_threshold": float(thresh_val), "confidence": confidence,
    }


def _percentile_centroid_from_deviation(deviation: np.ndarray, percentile: float = 98.0, min_area: int = 20) -> dict | None:
    """
    Thresholds at a fixed percentile of the deviation map's own pixel
    values (e.g. top 2%), rather than Otsu's assumption of a roughly
    balanced two-class split. Better suited to a small blob against a
    large, mostly-uniform background, which is the actual geometry here.
    """
    if deviation.dtype != np.uint8:
        deviation = np.clip(deviation, 0, 255).astype(np.uint8)
    thresh_val = float(np.percentile(deviation, percentile))
    mask = (deviation >= thresh_val).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < min_area:
        return None

    blob_mask = np.zeros_like(mask)
    cv2.drawContours(blob_mask, [largest], -1, 255, -1)
    M = cv2.moments(blob_mask, binaryImage=True)
    if not M["m00"]:
        return None
    return {"cx": float(M["m10"] / M["m00"]), "cy": float(M["m01"] / M["m00"]), "area": float(area), "threshold": thresh_val}


def _power_weighted_centroid_no_threshold(deviation: np.ndarray, power: float = 3.0) -> dict | None:
    """
    No segmentation step at all: a weighted centroid over the whole crop,
    with weights raised to a power so the strongest response dominates and
    background noise gets suppressed naturally, without ever picking a
    hard cutoff.
    """
    weights = deviation.astype(np.float64) ** power
    total = weights.sum()
    if total <= 0:
        return None
    ys, xs = np.indices(deviation.shape)
    return {"cx": float((xs * weights).sum() / total), "cy": float((ys * weights).sum() / total)}


def detect_dimple_from_delta_crop(delta_crop: np.ndarray, min_area: int = 20) -> dict | None:
    """delta_crop: as saved by build_dataset.py's _delta_crop (BGR or grayscale, 128 = no change)."""
    deviation = _deviation_from_delta_crop(delta_crop)
    return _weighted_centroid_from_deviation(deviation, min_area=min_area)


def detect_dimple_all_methods_from_delta_crop(delta_crop: np.ndarray, min_area: int = 20, power_sweep: tuple = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)) -> dict:
    """
    Runs every detector variant on the same decoded delta crop, so a
    comparison script doesn't need a separate multi-minute pass per method.
    Returns a dict of method_name -> result dict (or None if that method
    found nothing). Includes a sweep of power-weighting exponents, since
    that method won decisively and it's worth checking whether the
    original choice of power was actually the best one available.
    """
    deviation = _deviation_from_delta_crop(delta_crop)
    out = {
        "otsu": _weighted_centroid_from_deviation(deviation, min_area=min_area),
        "percentile_98": _percentile_centroid_from_deviation(deviation, percentile=98.0, min_area=min_area),
        "percentile_95": _percentile_centroid_from_deviation(deviation, percentile=95.0, min_area=min_area),
    }
    for p in power_sweep:
        out[f"power_{p:g}"] = _power_weighted_centroid_no_threshold(deviation, power=p)
    return out


def detect_dimple_from_pair(current_crop: np.ndarray, baseline_crop: np.ndarray, min_area: int = 20) -> dict | None:
    """Same idea as detect_dimple_from_delta_crop, computed directly from a current+baseline pair if you don't have a saved delta crop."""
    cur_gray = cv2.cvtColor(current_crop, cv2.COLOR_BGR2GRAY) if current_crop.ndim == 3 else current_crop
    base_gray = cv2.cvtColor(baseline_crop, cv2.COLOR_BGR2GRAY) if baseline_crop.ndim == 3 else baseline_crop
    deviation = cv2.absdiff(cur_gray, base_gray)
    return _weighted_centroid_from_deviation(deviation, min_area=min_area)


def fit_affine_mm_to_pixel(mm_pts: np.ndarray, px_pts: np.ndarray) -> dict:
    """Least-squares affine fit: pixel = A @ mm + b. Returns A, b, and the
    per-point residual (in pixels) so fit quality can be reported, not just
    assumed."""
    design = np.hstack([mm_pts, np.ones((len(mm_pts), 1))])
    params, *_ = np.linalg.lstsq(design, px_pts, rcond=None)
    A = params[:2, :].T
    b = params[2, :]
    pred_px = mm_pts @ A.T + b
    residuals = np.linalg.norm(pred_px - px_pts, axis=1)
    return {"A": A, "b": b, "residuals": residuals}


def extract_all_features(
    img_path: str | Path,
    roi: RoiConfig,
    baseline_img: np.ndarray | None = None,
) -> dict | None:
    """Convenience wrapper: load an image and extract both feature sets."""
    img = load_image(img_path)
    if img is None:
        return None

    fringe_crop = crop(img, roi.fringe_roi)
    indent_crop = crop(img, roi.indent_roi)

    baseline_fringe = baseline_indent = None
    if baseline_img is not None:
        baseline_fringe = crop(baseline_img, roi.fringe_roi)
        baseline_indent = crop(baseline_img, roi.indent_roi)

    feats = {}
    feats.update(extract_fringe_features(fringe_crop, baseline_fringe))
    feats.update(extract_indentation_features(indent_crop, baseline_indent))
    return feats
