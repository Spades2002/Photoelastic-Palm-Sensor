"""
OpenCV utilities for the mirror ROI: indenter centroid extraction and a
simple pixel -> physical (mm) mapping.

These also work stand-alone, independent of the PyTorch pipeline, e.g. for
a live calibration check or a quick demo.
"""
from __future__ import annotations

import cv2
import numpy as np


def crop_fractional_roi(image: np.ndarray, roi_frac: tuple) -> np.ndarray:
    """Crop an image using a region given as fractions of width and height.

    roi_frac: (x1, y1, x2, y2), each in [0, 1].
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = roi_frac
    px1, py1, px2, py2 = round(x1 * w), round(y1 * h), round(x2 * w), round(y2 * h)
    return image[py1:py2, px1:px2]


def compute_indenter_centroid(
    mirror_roi: np.ndarray,
    blur_ksize: int = 5,
    min_contour_area: float = 30.0,
) -> dict:
    """Locate the indenter tip reflected in the mirror ROI and return its centroid.

    Returns a dict with:
        centroid_uv : (u, v) centroid in ROI-local pixel coordinates, or None
        contour     : the selected contour, or None
        mask        : the binary mask used for contour detection

    centroid_uv and contour are None when nothing usable is found, so callers
    should check for that before using the result.
    """
    gray = cv2.cvtColor(mirror_roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # The indenter tip shows up as the brightest, most compact highlight in
    # the mirror reflection, so Otsu thresholding on the bright side works
    # well against a comparatively dark, uniform mirror background.
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"centroid_uv": None, "contour": None, "mask": mask}

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_contour_area:
        return {"centroid_uv": None, "contour": None, "mask": mask}

    moments = cv2.moments(largest)
    if moments["m00"] == 0:
        return {"centroid_uv": None, "contour": largest, "mask": mask}

    u = moments["m10"] / moments["m00"]
    v = moments["m01"] / moments["m00"]
    return {"centroid_uv": (u, v), "contour": largest, "mask": mask}


def roi_to_full_image_coords(u: float, v: float, roi_frac: tuple, image_shape: tuple) -> tuple:
    """Convert a centroid measured in ROI-local pixels back to full-image pixels."""
    h, w = image_shape[:2]
    x1, y1, _, _ = roi_frac
    return u + x1 * w, v + y1 * h


def pixel_to_physical_linear(u: float, v: float, calibration: dict) -> tuple:
    """Simple affine pixel -> mm mapping, one independent scale+offset per axis.

    calibration keys: origin_u, origin_v (pixel coordinates corresponding to
    x_mm=0, y_mm=0), scale_x, scale_y (mm per pixel, can be negative to flip
    an axis). Only correct if pixel space and physical space share the same
    orientation with no rotation or skew. For a data-driven fit that handles
    rotation and skew too, use fit_affine_calibration / apply_affine_calibration
    instead, via fit_calibration.py.
    """
    x_mm = (u - calibration["origin_u"]) * calibration["scale_x"]
    y_mm = (v - calibration["origin_v"]) * calibration["scale_y"]
    return x_mm, y_mm


def fit_affine_calibration(pixel_pts: np.ndarray, physical_pts: np.ndarray) -> dict:
    """Fit a full 2D affine pixel -> mm mapping by ordinary least squares over
    many paired (pixel centroid, true position) points, rather than a
    handful of manually measured calibration points. Unlike
    pixel_to_physical_linear, this handles any combination of scale,
    rotation and skew between pixel space and physical space, and gets more
    reliable the more points you fit it on. See fit_calibration.py, which
    builds pixel_pts/physical_pts directly from your existing dataset.

    pixel_pts: (N, 2) array of (u, v) pixel centroids, full-image coordinates.
    physical_pts: (N, 2) array of matching ground-truth (x_mm, y_mm).

    Returns {"A": 2x2 list, "b": 2-element list} such that
    [x_mm, y_mm] = A @ [u, v] + b
    """
    n = pixel_pts.shape[0]
    design = np.hstack([pixel_pts, np.ones((n, 1))])  # (N, 3): [u, v, 1]
    params, _, _, _ = np.linalg.lstsq(design, physical_pts, rcond=None)  # (3, 2)
    return {"A": params[:2, :].T.tolist(), "b": params[2, :].tolist()}


def apply_affine_calibration(u: float, v: float, calibration: dict) -> tuple:
    """Apply a calibration fitted by fit_affine_calibration to one (u, v) point."""
    A = np.array(calibration["A"])
    b = np.array(calibration["b"])
    x_mm, y_mm = A @ np.array([u, v]) + b
    return float(x_mm), float(y_mm)


def fit_pixel_to_physical_homography(pixel_pts: np.ndarray, physical_pts: np.ndarray) -> np.ndarray:
    """Fit a homography from four or more (pixel, physical mm) point pairs.

    Use this instead of pixel_to_physical_linear if the mirror view shows
    noticeable perspective distortion. Returns a 3x3 homography matrix;
    apply it with cv2.perspectiveTransform.
    """
    homography, _ = cv2.findHomography(pixel_pts, physical_pts, method=0)
    return homography


def compute_delta_crop(current_crop: np.ndarray, baseline_crop: np.ndarray) -> np.ndarray:
    """Compute a background-subtracted "delta" image from a current ROI crop
    and a no-contact baseline ROI crop of the same region, both single-channel.

    Formula confirmed empirically (not guessed): signed difference offset by
    128, i.e. delta = clip(current - baseline + 128, 0, 255). This preserves
    the direction of change (brighter-than-baseline vs darker-than-baseline)
    around a neutral mid-gray, rather than collapsing both directions into
    the same value the way absolute difference would. Verified against the
    real pipeline's fringe_delta_path/indent_delta_path output via
    diagnose_delta_formula.py: correlation 0.99 (fringe) once ROI alignment
    was corrected.
    """
    current = current_crop.astype(np.int16)
    baseline = baseline_crop.astype(np.int16)
    delta = (current - baseline + 128).clip(0, 255)
    return delta.astype(np.uint8)


def visualise_rois(image_path: str, green_roi_frac: tuple, blue_roi_frac: tuple, save_path: str = None):
    """Draw the configured ROIs on a sample frame, to help tune the fractions
    in config.py before committing to a full training run. Opens a window
    unless save_path is given, in which case the annotated frame is written
    to disk instead.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h, w = image.shape[:2]
    vis = image.copy()

    for roi_frac, colour in ((green_roi_frac, (0, 255, 0)), (blue_roi_frac, (255, 0, 0))):
        x1, y1, x2, y2 = roi_frac
        pt1 = (int(x1 * w), int(y1 * h))
        pt2 = (int(x2 * w), int(y2 * h))
        cv2.rectangle(vis, pt1, pt2, colour, 2)

    if save_path:
        cv2.imwrite(save_path, vis)
    else:
        cv2.imshow("ROI preview, press any key to close", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
