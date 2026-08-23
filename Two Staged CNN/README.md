# Photoelastic Tactile Sensor: Physics-Informed Two-Stage Force Estimation

Deep learning pipeline for estimating **indentation depth**, **contact location** and **contact force** from differential camera images captured by a photoelastic tactile sensor.

The architecture separates visual estimation of indentation depth and contact position from the subsequent force-regression stage.

Two independent convolutional branches first estimate:

- Indentation depth, \(d\), from the photoelastic fringe delta crop
- Contact position, \((x,y)\), from the mirror delta crop

These predictions are then combined and supplied to a multi-layer perceptron that estimates the contact-force vector.

The complete prediction pipeline is:

```text id="pq0hw3"
Fringe delta crop ──> DepthCNN ──────────────> depth_mm ────┐
                                                           │
                                                           ├──> ForceMLP
                                                           │       │
Mirror delta crop ──> LocalisationCNN ──> x_mm, y_mm ─────┘       │
                                                                   ▼
                                                        Fx, Fy, Fz, |F|
```

Force estimation is therefore performed from the **estimated physical quantities** of indentation depth and contact position rather than directly from learned image features.

## Model Architecture

The architecture consists of a `DepthCNN`, `LocalisationCNN` and `ForceMLP`.

### DepthCNN

The `DepthCNN` estimates indentation depth from the photoelastic fringe delta image:

```text id="flp1v4"
Fringe delta crop
        │
        ▼
Single-channel ResNet-18
        │
        ▼
   depth_mm
```

A pretrained ResNet-18 is adapted to accept single-channel greyscale images.

The original RGB input convolution is replaced with a one-channel convolution. The pretrained RGB filters are averaged across their channel dimension to initialise the new greyscale convolution, allowing the network to retain the pretrained low-level feature representation.

The final ResNet fully connected layer is replaced with a single regression output:

```text id="xfn6zo"
DepthCNN(fringe_delta) -> depth_mm
```

### LocalisationCNN

The `LocalisationCNN` has the same general ResNet-18 architecture but operates on the mirror delta crop:

```text id="sdmdxa"
Mirror delta crop
        │
        ▼
Single-channel ResNet-18
        │
        ▼
    x_mm, y_mm
```

Its final regression layer contains two outputs corresponding to the estimated two-dimensional contact position.

The localisation branch does not receive the predicted indentation depth as an input. The two CNN branches therefore perform their visual predictions separately before their outputs are combined for force estimation.

## Single-Channel Greyscale Input

Unlike the RGB-based architectures, both CNN branches operate on **single-channel greyscale differential images**.

Each delta crop is loaded as a greyscale image and resized to:

```text id="j8w6qe"
224 × 224
```

The image is then converted to a floating-point tensor with shape:

```text id="i5h5w0"
1 × 224 × 224
```

and normalised to the range `[0, 1]`.

The two network inputs are therefore:

```text id="nb6k31"
DepthCNN:
    fringe delta crop
    1 × 224 × 224

LocalisationCNN:
    mirror delta crop
    1 × 224 × 224
```

This allows the networks to operate directly on spatial variations in greyscale intensity within the photoelastic fringe and mirror regions without requiring RGB colour information.

## Differential Image Representation

Both CNNs operate on differential rather than raw sensor images.

For a loaded image \(I\) and corresponding unloaded reference image \(I_0\), the differential representation isolates changes associated with sensor deformation.

Conceptually:

```text id="f4l97w"
Loaded image
     │
     ├── Unloaded baseline
     │
     ▼
Differential image
```

The pipeline uses separate differential regions for the two prediction tasks:

```text id="c4a7uf"
Fringe delta  ──> indentation depth

Mirror delta  ──> contact location
```

The unloaded baseline corresponds to the no-contact sensor state for the relevant measurement configuration.

## ForceMLP

The estimated indentation depth and contact position are concatenated to form a compact three-dimensional physical representation:

```text id="9yjpm6"
z = [depth_mm, x_mm, y_mm]
```

This representation is supplied to the `ForceMLP`.

The MLP architecture is:

```text id="exwq44"
depth_mm ─┐
          │
x_mm ─────┼──> Concatenate
          │         │
y_mm ─────┘         ▼
              3 input features
                     │
                     ▼
                Linear: 64
                   ReLU
                 Dropout
                     │
                     ▼
                Linear: 64
                   ReLU
                 Dropout
                     │
                     ▼
                Linear: 4
                     │
                     ▼
              Fx, Fy, Fz, |F|
```

The default hidden-layer configuration is:

```python id="vlaz2p"
hidden_dims = (64, 64)
dropout = 0.1
```

The force stage therefore does not receive the original images or ResNet feature vectors directly.

Instead:

```text id="a8odfh"
Images
   │
   ▼
Estimated physical quantities
   │
   ▼
[depth_mm, x_mm, y_mm]
   │
   ▼
ForceMLP
   │
   ▼
[Fx, Fy, Fz, |F|]
```

This explicitly separates visual perception from the subsequent mapping between estimated contact state and force.

## End-to-End Training

Although the depth and localisation CNNs perform separate visual prediction tasks, the complete architecture is trained jointly.

For every sample, the model predicts:

```text id="kk8nd1"
depth_pred
xy_pred
force_pred
```

These are compared against the corresponding ground-truth measurements:

```text id="w6k8n7"
depth_mm
x_mm, y_mm
Fx, Fy, Fz, |F|
```

All downstream calculations use **predicted** depth and location.

Ground-truth depth and location values are used only as supervision targets during training and are never supplied to the `ForceMLP` as inputs.

This distinction is important because, during deployment, indentation depth and contact position are not known beforehand.

## Training Objective

The prediction components are supervised using the **Huber loss**.

Three loss terms are calculated:

```text id="z8qpxr"
L_depth
L_location
L_force
```

The total objective is:

```text id="o5kwh1"
L_total =
    λ_depth    L_depth
  + λ_location L_location
  + λ_force    L_force
```

The default configuration uses:

```text id="twepm3"
λ_depth    = 1.0
λ_location = 1.0
λ_force    = 1.0
```

giving:

```text id="3w4jv7"
L_total = L_depth + L_location + L_force
```

with a Huber delta of:

```text id="i2z4oa"
δ = 1.0
```

Because the `ForceMLP` receives the predicted depth and contact position, the force loss can propagate through the MLP into the preceding CNN branches during end-to-end training.

The complete architecture is therefore optimised jointly while retaining separate depth and localisation prediction pathways.

## Information Flow

The complete model can be represented as:

```text id="jrz2of"
                  Sensor image
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
      Fringe delta crop    Mirror delta crop
             │                   │
             ▼                   ▼
         DepthCNN         LocalisationCNN
        ResNet-18            ResNet-18
             │                   │
             ▼                   ▼
         depth_mm            x_mm, y_mm
             │                   │
             └─────────┬─────────┘
                       │
                       ▼
          [depth_mm, x_mm, y_mm]
                       │
                       ▼
                   ForceMLP
                       │
                       ▼
              Fx, Fy, Fz, |F|
```

This creates an interpretable intermediate representation between the image-processing and force-regression stages.

## Repository Structure

```text id="ix4h7w"
.
├── config.py
├── cv_utils.py
├── dataset.py
├── model.py
├── train.py
├── evaluate_only.py
├── predict_random_samples.py
├── predict_random_raw_samples.py
├── predict_visual.py
├── preview_rois.py
├── search_roi_offset.py
├── requirements.txt
└── README.md
```

### Core Files

| File | Description |
|---|---|
| `config.py` | Dataset paths, image parameters, training hyperparameters and ROI configuration. |
| `dataset.py` | PyTorch dataset for loading the fringe and mirror differential crops and their corresponding labels. |
| `model.py` | Defines the `DepthCNN`, `LocalisationCNN`, `ForceMLP` and complete `TactileForceNet`. |
| `train.py` | End-to-end training, validation, checkpointing and test evaluation. |
| `evaluate_only.py` | Re-evaluates an existing checkpoint without retraining. |
| `predict_random_samples.py` | Performs predictions on randomly selected held-out samples. |
| `predict_random_raw_samples.py` | Tests the complete raw-image preprocessing and inference pipeline. |
| `predict_visual.py` | Runs inference on an individual raw sensor image. |
| `preview_rois.py` | Visualises the configured fringe and mirror regions. |
| `search_roi_offset.py` | Utility for identifying the fixed ROI positions in the raw sensor image. |
| `cv_utils.py` | Image cropping, differential-image and supporting computer-vision utilities. |

## Installation

Clone the repository:

```bash id="7v5m3x"
git clone <repository-url>
cd <repository-name>
```

Install the required dependencies:

```bash id="1w8z2f"
pip install -r requirements.txt
```

The main dependencies include:

```text id="8fc7hd"
PyTorch
torchvision
OpenCV
NumPy
pandas
Matplotlib
```

A CUDA-compatible GPU can be used automatically when available.

## Dataset Configuration

Update the dataset paths in `config.py`:

```python id="yk6u2v"
CSV_PATH = Path("path/to/dataset.csv")
IMAGE_ROOT = Path("path/to/crops")
OUTPUT_DIR = Path("path/to/output")
```

The dataset CSV must contain paths to the two differential image crops:

```text id="xbw47a"
fringe_delta_path
indent_delta_path
```

along with the corresponding ground-truth measurements:

```text id="j3nt6v"
depth_mm
x_mm
y_mm
Fx
Fy
Fz
```

The force magnitude is calculated from the measured force components for use as an additional regression target.

## Training

Train the complete model using:

```bash id="d33r09"
python train.py
```

The training pipeline jointly optimises the `DepthCNN`, `LocalisationCNN` and `ForceMLP`.

The default training configuration includes:

```text id="v89i6a"
Batch size:       32
Learning rate:    1e-4
Weight decay:     1e-5
Maximum epochs:   100
Huber delta:      1.0
```

Early stopping is applied using validation loss, with a default patience of 15 epochs.

The best-performing model is saved to:

```text id="1ce6vk"
checkpoints/best_model.pt
```

## Evaluation

A trained checkpoint can be evaluated without retraining using:

```bash id="dgnqqf"
python evaluate_only.py
```

The evaluation pipeline reports predictions for:

```text id="j5w4tb"
Indentation depth:
    depth_mm

Contact location:
    x_mm
    y_mm

Contact force:
    Fx
    Fy
    Fz
    magnitude
```

The predictions are saved under the configured output directory for subsequent quantitative analysis.

## Inference

The repository provides two inference routes.

For already processed differential crops:

```bash id="v4zcr9"
python predict_random_samples.py
```

For raw sensor images, the pipeline can perform ROI extraction and differential preprocessing before inference:

```bash id="ks5kgx"
python predict_visual.py --image "path/to/raw/photo.jpg"
```

The raw-image pipeline extracts the fixed fringe and mirror regions and compares them with the corresponding unloaded baseline before passing the resulting greyscale delta images to the model.

## Method Summary

The central distinction of this architecture is the use of **estimated physical quantities as intermediate features**.

Rather than learning a direct mapping:

```text id="bg3bkp"
image features ──> force
```

the model learns:

```text id="4dgy40"
fringe image ──> indentation depth ──┐
                                    ├──> force
mirror image ──> contact position ──┘
```

The approach therefore decomposes tactile inference into interpretable intermediate predictions before estimating the final force components.

## Citation

If you use this repository or build upon this work in academic research, please cite the associated publication or dissertation.

```bibtex id="jofn2h"
@misc{photoelastic_tactile_two_stage,
  title  = {Physics-Informed Two-Stage Photoelastic Tactile Force Estimation},
  author = {Staines Rajith},
  year   = {2026}
}
```

## Licence

This repository is provided for research and educational use. Add an appropriate licence file before redistribution or external reuse.