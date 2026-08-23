# Photoelastic Tactile Sensor: Force and Contact Localisation

Deep learning pipeline for estimating **contact force** and **contact location** from camera images captured by a photoelastic tactile sensor.

The model uses a dual-branch convolutional architecture based on pretrained **ResNet-18** networks. Raw image crops of the photoelastic fringe region and the indenter-contact mirror region are processed in parallel to predict the applied force and corresponding contact position.

## Model Architecture

The network consists of two pretrained ResNet-18 backbones operating in parallel:

- **Fringe branch:** receives the raw photoelastic fringe crop, \(I_{\text{fringe}}\).
- **Mirror branch:** receives the raw indenter-contact mirror crop, \(I_{\text{mirror}}\).

Each ResNet-18 produces a 512-dimensional feature embedding. The two embeddings are concatenated to form a joint 1024-dimensional representation.

```text
Raw fringe crop ──> ResNet-18 ──┐
                                │
                                ├──> Concatenated features ──> Force head ──> Fx, Fy, Fz
                                │
Raw mirror crop ──> ResNet-18 ──┘
                                                    │
                                                    └──> Localisation head ──> x, y
```

The force and localisation heads operate independently from the same concatenated feature representation. The output of the force head is not supplied to the localisation head.

The model therefore predicts:

```text
Force:      Fx, Fy, Fz
Location:   x_mm, y_mm
```

Both prediction tasks are trained jointly using the corresponding ground-truth force and contact-position measurements.

## Input Images

Two regions of interest are extracted from each raw camera frame.

### Photoelastic Fringe Region

The fringe crop contains the stress-induced photoelastic patterns generated when a load is applied to the sensor surface. This region provides visual information associated with the applied contact force.

### Indenter-Contact Mirror Region

The mirror crop contains the reflected view of the sensor deformation and indenter contact. This provides spatial information for estimating the contact position on the sensor surface.

This architecture operates on **raw, non-differential image crops**. No unloaded reference image or baseline subtraction is required for inference.

Both crops are resized to `224 × 224` pixels and normalised using ImageNet statistics before being passed to their respective ResNet-18 backbones.

## Repository Structure

```text
.
├── config.py
├── dataset_index.py
├── feature_extraction.py
├── build_dataset.py
├── train_resnet.py
├── train.py
├── evaluate_resnet.py
├── predict.py
├── requirements.txt
└── README.md
```

### Files

| File | Description |
|---|---|
| `config.py` | Defines dataset paths, output directories and ROI configuration. |
| `dataset_index.py` | Links force measurements with the corresponding image folders and experimental metadata. |
| `feature_extraction.py` | Handles image loading and extraction of the fringe and mirror regions of interest. |
| `build_dataset.py` | Constructs the training dataset and saves the required image crops. |
| `train_resnet.py` | Contains the dual-ResNet model, dataset class and training pipeline. |
| `train.py` | Entry point configured for the raw-crop, independent-head architecture. |
| `evaluate_resnet.py` | Evaluates the saved model checkpoint on held-out contact locations. |
| `predict.py` | Runs inference on a new tactile-sensor image. |
| `requirements.txt` | Python dependencies required by the pipeline. |

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

Create and activate a virtual environment if required, then install the dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies include PyTorch, torchvision, OpenCV, NumPy, pandas, scikit-learn and Pillow.

For GPU training, install the appropriate CUDA-enabled version of PyTorch for your system.

## Dataset Configuration

Before running the pipeline, update the paths in `config.py`:

```python
DATASET_ROOT = Path("path/to/dataset")
IMAGES_ROOT = DATASET_ROOT / "images"

OUTPUT_ROOT = Path("path/to/output")
```

The experimental dataset is expected to contain force measurements and corresponding image folders for each indentation state.

Ground-truth targets are:

```text
Fx
Fy
Fz
x_mm
y_mm
```

Multiple images recorded while the indenter is held at the same position and indentation depth correspond to the same physical contact state and therefore share the same force and location labels.

## Building the Dataset

Generate the dataset and raw image crops using:

```bash
python build_dataset.py --save-crops --crop-types raw --phases down
```

This produces the processed `dataset.csv` together with the raw fringe and mirror crops required for training.

The generated dataset contains paths to the image crops alongside the corresponding force and contact-location ground truth.

## Training

Train the raw-crop model using:

```bash
python train.py --epochs 30 --batch-size 32
```

The training entry point configures the model as:

```python
crop_source="raw"
force_targets=["Fx", "Fy", "Fz"]
force_conditions_location=False
```

This ensures that the implementation corresponds to the raw-image, independent-head architecture.

The two ResNet-18 backbones are initialised using pretrained ImageNet weights and trained jointly with the force and localisation heads.

The overall training objective combines the force and localisation regression losses:

```text
Loss = MSE(force prediction, force ground truth)
     + MSE(location prediction, location ground truth)
```

Training uses a spatially grouped train-validation split based on physical contact locations. Images belonging to a held-out sensor location therefore do not appear in the training set, reducing spatial data leakage between the two sets.

The best-performing model checkpoint is saved to the configured models directory.

## Evaluation

Evaluate the trained checkpoint using:

```bash
python evaluate_resnet.py
```

The evaluation script reconstructs the same held-out spatial split used during training and evaluates the checkpoint saved during the best validation epoch.

The model is evaluated separately for force and contact localisation.

## Inference

A trained model can be used to estimate force and contact position from a new sensor image:

```bash
python predict.py --image path/to/photo.png --model resnet
```

For the raw-crop architecture, a baseline image is not required.

Example output:

```text
{
    "Fx": ...,
    "Fy": ...,
    "Fz": ...,
    "x_mm": ...,
    "y_mm": ...
}
```

## Method Summary

For an input camera image, the inference pipeline is:

```text
Camera image
     │
     ├──> Raw fringe ROI ──> ResNet-18 ──┐
     │                                   │
     │                                   ├──> Feature concatenation
     │                                   │             │
     └──> Raw mirror ROI ──> ResNet-18 ──┘             │
                                                       ├──> Force head ──> Fx, Fy, Fz
                                                       │
                                                       └──> Location head ──> x, y
```

This architecture enables simultaneous estimation of contact force and spatial contact location from the visual response of the photoelastic tactile sensor.

## Citation

If you use this repository or build upon this work in academic research, please cite the associated publication or dissertation.

```bibtex
@misc{photoelastic_tactile_sensor,
  title  = {Photoelastic Tactile Sensing for Force Estimation and Contact Localisation},
  author = {Staines Rajith},
  year   = {2026}
}
```

## Licence

This repository is provided for research and educational use. Add an appropriate licence file to the repository before redistribution or external reuse.