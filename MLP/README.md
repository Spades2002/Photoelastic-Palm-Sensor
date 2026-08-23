# Photoelastic Tactile Sensor: MLP Force Regression from Differential Fringe Intensities

Machine learning pipeline for estimating **three-axis contact force** from differential photoelastic fringe images captured by a tactile sensor.

Unlike the convolutional architectures, this method does not use a CNN or preserve the fringe crop as a two-dimensional image during regression. Instead, the differential fringe crop is converted to greyscale, resized and flattened into a one-dimensional pixel-intensity vector. This vector is supplied directly to a **multi-layer perceptron regressor (MLP)** implemented using scikit-learn's `MLPRegressor`.

The model predicts:

```text id="c32z92"
Force: Fx, Fy, Fz
```

This architecture performs **force estimation only** and does not predict contact location.

## Model Architecture

The pipeline uses only the photoelastic fringe region of the sensor image.

```text id="14od0q"
Loaded sensor image
        │
        ▼
  Fringe ROI crop
        │
        ▼
Greyscale conversion
        │
        ▼
 Resize to 64 × 64
        │
        ▼
Subtract unloaded baseline
        │
        ▼
Differential fringe vector
        │
        ▼
      Flatten
        │
        ▼
4,096 pixel-intensity values
        │
        ▼
  StandardScaler
        │
        ▼
 MLPRegressor
        │
        ▼
   Fx, Fy, Fz
```

No convolutional layers or ResNet feature extractors are used.

Instead, the flattened differential pixel intensities are supplied directly to the regression network.

## Differential Fringe Representation

The model operates on the change in the photoelastic fringe region between the loaded and unloaded sensor states.

For each image, the configured active sensor region is first extracted, followed by the photoelastic fringe region:

```text id="oy5ebz"
Raw image
    │
    ▼
Active sensor ROI
    │
    ▼
Photoelastic fringe ROI
```

The fringe crop is then converted to greyscale and resized to:

```text id="knv9qq"
64 × 64 pixels
```

The resized image is normalised to the range `[0, 1]` and flattened into a one-dimensional vector:

```text id="evl0m5"
64 × 64 image
      │
      ▼
    Flatten
      │
      ▼
4,096-element vector
```

A corresponding unloaded baseline is selected for each sensor grid position and indentation cycle.

The differential representation is then calculated as:

```text id="hwc48j"
Δx = x_loaded - x_baseline
```

where `x_loaded` is the flattened greyscale fringe vector for the loaded sensor state and `x_baseline` is the corresponding unloaded reference vector.

The resulting differential vector therefore represents the change in fringe intensity caused by loading rather than the absolute appearance of the sensor.

## Why Use a Flattened Representation?

The purpose of this architecture is to evaluate whether contact force can be recovered from a simple pixel-intensity representation without convolutional feature extraction.

The preceding ResNet-based architectures preserve the two-dimensional image structure and learn spatial features through convolution.

This model instead represents each fringe image as:

```text id="10gqrx"
Δx = [p₁, p₂, p₃, ..., p₄₀₉₆]
```

where each \(p_i\) represents the differential intensity of one pixel in the resized fringe crop.

The MLP therefore does not explicitly exploit two-dimensional spatial locality or translation-aware convolutional features within the photoelastic fringe pattern.

This provides a non-convolutional baseline against which the image-based ResNet architectures can be compared.

## MLP Architecture

The regression model is implemented using scikit-learn's `MLPRegressor`.

Before entering the MLP, the 4,096 input features are standardised using `StandardScaler`.

The network architecture is:

```text id="gtgqpe"
4,096 input features
        │
        ▼
 StandardScaler
        │
        ▼
Dense layer: 256
      ReLU
        │
        ▼
Dense layer: 128
      ReLU
        │
        ▼
Dense layer: 64
      ReLU
        │
        ▼
3 regression outputs
        │
        ▼
   Fx, Fy, Fz
```

The principal MLP configuration is:

```python id="8a1cd5"
MLPRegressor(
    hidden_layer_sizes=(256, 128, 64),
    activation="relu",
    alpha=1e-4,
    learning_rate_init=1e-3,
    max_iter=2000,
    early_stopping=True,
    n_iter_no_change=20,
    random_state=42,
)
```

The model is wrapped in a scikit-learn pipeline:

```python id="oaqhnd"
model = make_pipeline(
    StandardScaler(),
    MLPRegressor(...)
)
```

This ensures that the flattened pixel-intensity features are standardised before being supplied to the neural network.

## Force Prediction

The model performs multi-output regression to predict all three measured force components simultaneously:

```text id="5aguj3"
Input:
    Differential fringe vector

Output:
    Fx
    Fy
    Fz
```

The target vector for each training sample is therefore:

```text id="0v3r29"
y = [Fx, Fy, Fz]
```

Unlike the preceding dual-branch ResNet architectures, this model does not use the mirror or indenter-contact region and does not estimate the contact coordinates `(x_mm, y_mm)`.

Its sole objective is three-axis force regression from the differential photoelastic fringe response.

## Repository Structure

```text id="o0zgtc"
.
├── common.py
├── cv_utils.py
├── check_roi.py
├── build_features.py
├── train_mlp.py
├── requirements.txt
└── README.md
```

### Files

| File | Description |
|---|---|
| `common.py` | Contains shared configuration, dataset paths, ROI definitions, feature-cache loading and evaluation functions. |
| `cv_utils.py` | Provides image cropping and ROI visualisation utilities. |
| `check_roi.py` | Visualises the configured sensor and fringe ROIs and previews the image representation used by the model. |
| `build_features.py` | Builds the dataset manifest and generates the flattened differential fringe vectors. |
| `train_mlp.py` | Defines, trains and evaluates the `MLPRegressor`. |
| `requirements.txt` | Python dependencies required by the pipeline. |

## Installation

Clone the repository:

```bash id="4m0lgz"
git clone <repository-url>
cd <repository-name>
```

Install the dependencies:

```bash id="7l5elr"
pip install -r requirements.txt
```

The main dependencies are:

```text id="1arssz"
numpy
pandas
opencv-python
scikit-learn
joblib
```

The pipeline uses scikit-learn's `MLPRegressor` and therefore runs on the CPU.

## Dataset Configuration

Before processing the dataset, edit `CONFIG` in `common.py`:

```python id="g43o2x"
CONFIG = {
    "root_dir": "path/to/dataset",
    "active_roi_frac": (...),
    "fringe_roi_frac": (...),
    "flatten_size": (64, 64),
    "val_fraction": 0.15,
    "seed": 42,
    "output_dir": "path/to/output",
}
```

The important image-processing parameters are:

```text id="97aljw"
active_roi_frac
fringe_roi_frac
flatten_size
```

The model uses only the fringe region for force prediction.

## Checking the Fringe ROI

Before processing the complete dataset, the configured ROI can be checked using:

```bash id="vp2ekc"
python check_roi.py "path/to/sample/frame.jpg"
```

This generates previews showing the active sensor region and the selected fringe region.

It also saves a preview of the greyscale `64 × 64` representation used to construct the flattened model input.

This step is useful for ensuring that the configured crop captures the photoelastic fringe response before feature generation.

## Building the Differential Features

Generate the flattened feature arrays using:

```bash id="4klgpd"
python build_features.py
```

For each sensor image, the script:

1. extracts the configured fringe region,
2. converts the crop to greyscale,
3. resizes it to `64 × 64`,
4. normalises the pixel intensities,
5. flattens the image into a 4,096-element vector,
6. subtracts the corresponding unloaded baseline vector.

The resulting differential features are cached to:

```text id="3h9p2h"
features_delta.npz
```

The cache contains:

```text id="g9x5j0"
X_train
X_val
y_train
y_val
```

where `X` contains the flattened differential fringe vectors and `y` contains the corresponding three-axis force measurements.

## Train-Validation Split

The dataset is split using a **group-aware spatial split**.

Samples are grouped according to their physical sensor grid position:

```text id="mrzrzf"
(row, col)
```

All images associated with a particular grid point are assigned entirely to either the training or validation set.

This prevents images from the same physical contact location appearing in both sets and reduces spatial data leakage during evaluation.

The default validation fraction is:

```text id="iz2ccv"
15%
```

with a random seed of:

```text id="g9ebx1"
42
```

## Training

Train the MLP using:

```bash id="brs3gw"
python train_mlp.py
```

The model receives the flattened differential fringe vector and learns to regress:

```text id="htjj3s"
[Fx, Fy, Fz]
```

Early stopping is enabled. Training stops when the internal validation score has not improved for 20 iterations, up to a maximum of 2,000 iterations.

## Evaluation

Performance is evaluated independently for each force component using:

```text id="5k6a3m"
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
Coefficient of Determination (R²)
```

Overall MAE and \(R^2\) are also calculated across the three force components.

The trained model can be saved using `joblib`, allowing it to be loaded again without retraining.

## Method Summary

The complete pipeline is:

```text id="ijap96"
                 Sensor image
                      │
                      ▼
               Active sensor ROI
                      │
                      ▼
                 Fringe ROI
                      │
                      ▼
                  Greyscale
                      │
                      ▼
                  64 × 64
                      │
                      ▼
                   Flatten
                      │
                      ▼
              4,096-D vector
                      │
            subtract baseline
                      │
                      ▼
         Differential intensity vector
                      │
                      ▼
               StandardScaler
                      │
                      ▼
                  MLPRegressor
                      │
              ┌───────┼───────┐
              ▼       ▼       ▼
             Fx      Fy      Fz
```

This architecture provides a purely non-convolutional force-regression approach based on flattened differential fringe intensities, allowing its performance to be compared with spatial feature representations learned by the ResNet-based models.

## Citation

If you use this repository or build upon this work in academic research, please cite the associated publication or dissertation.

```bibtex id="rr5x4t"
@misc{photoelastic_tactile_mlp,
  title  = {Intensity-Based Photoelastic Tactile Force Estimation Using Multi-Layer Perceptron Regression},
  author = {Staines Rajith},
  year   = {2026}
}
```

## Licence

This repository is provided for research and educational use. Add an appropriate licence file before redistribution or external reuse.