"""
Run a single new photo through the pipeline and print an estimated force
and location. Works with either the linear baseline (default, no extra
dependencies) or the ResNet model (--model resnet, needs torch/torchvision
and a checkpoint from train_resnet.py).

--baseline-image: required for the linear model to get its best (diff-based)
features, and required for the resnet model if that checkpoint was trained
with --crop-source delta (the default since the delta-crop change), since
the model needs the same baseline-subtracted crop it was trained on. Not
needed for an older resnet checkpoint trained with --crop-source raw.

Usage:
    python predict.py --image path/to/photo.png
    python predict.py --image path/to/photo.png --baseline-image path/to/0mm_photo.png --model resnet
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np

import config
import feature_extraction as fe
from train_linear import FRINGE_FEATURE_COLS, INDENT_FEATURE_COLS


def predict_linear(image_path: str, baseline_image_path: str | None = None):
    roi = fe.RoiConfig.load()
    baseline_img = fe.load_image(baseline_image_path) if baseline_image_path else None
    feats = fe.extract_all_features(image_path, roi, baseline_img=baseline_img)
    if feats is None:
        raise SystemExit(f"Could not read image: {image_path}")

    force_model = joblib.load(config.MODELS_ROOT / "linear_force_model.joblib")
    location_model = joblib.load(config.MODELS_ROOT / "linear_location_model.joblib")

    X_force = np.array([[feats[c] for c in FRINGE_FEATURE_COLS]])
    X_loc = np.array([[feats[c] for c in INDENT_FEATURE_COLS]])

    Fx, Fy, Fz = force_model.predict(X_force)[0]
    x_mm, y_mm = location_model.predict(X_loc)[0]
    return {"Fx": float(Fx), "Fy": float(Fy), "Fz": float(Fz), "x_mm": float(x_mm), "y_mm": float(y_mm)}


def predict_resnet(image_path: str, baseline_image_path: str | None = None):
    import torch
    from PIL import Image
    from torchvision import transforms
    import cv2
    from train_resnet import load_tactile_checkpoint, IMAGENET_MEAN, IMAGENET_STD, LOCATION_TARGETS
    from build_dataset import _delta_crop

    roi = fe.RoiConfig.load()
    img = fe.load_image(image_path)
    if img is None:
        raise SystemExit(f"Could not read image: {image_path}")

    device = torch.device("cpu")
    model, ckpt = load_tactile_checkpoint(config.MODELS_ROOT / "resnet_tactile_best.pt", device)
    force_targets = ckpt.get("force_targets", ["Fx", "Fy", "Fz"])
    crop_source = ckpt.get("crop_source", "raw")

    fringe_crop = fe.crop(img, roi.fringe_roi)
    indent_crop = fe.crop(img, roi.indent_roi)

    if crop_source == "delta":
        if not baseline_image_path:
            raise SystemExit(
                "This checkpoint was trained on delta crops (baseline-subtracted), "
                "so --baseline-image is required, not optional, for the resnet model."
            )
        baseline_img = fe.load_image(baseline_image_path)
        if baseline_img is None:
            raise SystemExit(f"Could not read baseline image: {baseline_image_path}")
        fringe_crop = _delta_crop(fringe_crop, fe.crop(baseline_img, roi.fringe_roi))
        indent_crop = _delta_crop(indent_crop, fe.crop(baseline_img, roi.indent_roi))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    def to_tensor(bgr_crop):
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        return transform(Image.fromarray(rgb)).unsqueeze(0)

    with torch.no_grad():
        force_pred, loc_pred = model(to_tensor(fringe_crop), to_tensor(indent_crop))

    force = force_pred.numpy()[0] * ckpt["force_std"] + ckpt["force_mean"]
    loc = loc_pred.numpy()[0] * ckpt["loc_std"] + ckpt["loc_mean"]
    result = {t: float(force[i]) for i, t in enumerate(force_targets)}
    result.update({t: float(loc[i]) for i, t in enumerate(LOCATION_TARGETS)})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--baseline-image", default=None, help="Matching 0mm baseline photo, improves accuracy if given")
    parser.add_argument("--model", choices=["linear", "resnet"], default="linear")
    args = parser.parse_args()

    if args.model == "linear":
        result = predict_linear(args.image, args.baseline_image)
    else:
        result = predict_resnet(args.image, args.baseline_image)

    print(result)
