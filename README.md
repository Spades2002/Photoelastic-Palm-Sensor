# Photoelastic Tactile Sensing: Force Estimation and Contact Localisation

This repository contains the machine learning methods developed for a **camera-based photoelastic tactile sensor** capable of estimating contact force and contact location from visual stress patterns.

When contact is applied to the sensor, the photoelastic sensing layer produces stress-dependent fringe patterns that are captured by a camera. A secondary mirror view provides additional information about the contact region. The methods in this repository investigate different approaches for extracting force, indentation depth and contact-location information from these images.

Four learning-based architectures are included, ranging from end-to-end convolutional models to intensity-based and physics-informed approaches.

## Methods

### 1. Raw-Crop Dual-ResNet

A dual-branch architecture using two pretrained ResNet-18 networks operating on **raw, non-differential image crops**.

- The fringe branch processes the raw photoelastic fringe crop.
- The mirror branch processes the raw indenter-contact mirror crop.
- The resulting feature embeddings are concatenated.
- Independent regression heads estimate contact force and contact location.

**Outputs:** \(F_x, F_y, F_z, x, y\)

See the README inside the corresponding folder for implementation and training details.

---

### 2. Differential-Crop Force-Conditioned ResNet

A dual-branch ResNet-18 architecture operating on **differential fringe and mirror crops**, generated relative to the unloaded sensor state.

The model estimates normal force \(F_z\), which is subsequently used as an additional conditioning variable for contact localisation. The force prediction is detached before entering the localisation head, preventing the localisation loss from propagating through the force prediction.

**Outputs:** \(F_z, x, y\)

See the README inside the corresponding folder for implementation and training details.

---

### 3. Differential Intensity MLP

A non-convolutional approach designed to investigate whether force information can be recovered directly from the **pixel-intensity representation** of the photoelastic fringe response.

The greyscale fringe delta crop is resized and flattened into a one-dimensional intensity vector before being supplied to a scikit-learn `MLPRegressor`.

Unlike the image-based architectures, this method does not explicitly learn convolutional spatial features and performs force estimation only.

**Outputs:** \(F_x, F_y, F_z\)

See the README inside the corresponding folder for implementation and training details.

---

### 4. Physics-Informed Two-Stage CNN

A two-stage architecture that separates visual estimation of the sensor's physical contact state from subsequent force estimation.

A single-channel ResNet-18 predicts **indentation depth** from the fringe delta crop, while a second independent single-channel ResNet-18 predicts **contact location** from the mirror delta crop.

The predicted depth and contact position are then concatenated and supplied to a small MLP for force regression:

```text
Fringe delta ──> DepthCNN ─────────────> depth ─────┐
                                                    │
                                                    ├──> ForceMLP ──> Force
                                                    │
Mirror delta ──> LocalisationCNN ──> (x, y) ───────┘
```

**Outputs:** indentation depth, \(x, y, F_x, F_y, F_z\) and force magnitude

See the README inside the corresponding folder for implementation and training details.

## Repository Structure

```text
.
├── raw_crop_resnet/
│   ├── README.md
│   └── ...
│
├── differential_crop_resnet/
│   ├── README.md
│   └── ...
│
├── intensity_mlp/
│   ├── README.md
│   └── ...
│
├── physics_informed_two_stage/
│   ├── README.md
│   └── ...
│
└── README.md
```

Each method is self-contained within its own folder and includes a dedicated README describing its architecture, preprocessing, training procedure and usage.

## Overview

| Method | Image Input | Model | Force | Location |
|---|---|---|---|---|
| Raw-Crop Dual-ResNet | Raw RGB fringe + mirror | Dual ResNet-18 | \(F_x,F_y,F_z\) | \(x,y\) |
| Differential Force-Conditioned ResNet | Delta RGB fringe + mirror | Dual ResNet-18 | \(F_z\) | \(x,y\) |
| Differential Intensity MLP | Greyscale fringe delta | MLPRegressor | \(F_x,F_y,F_z\) | No |
| Physics-Informed Two-Stage CNN | Greyscale fringe + mirror delta | Dual ResNet-18 + MLP | \(F_x,F_y,F_z,|F|\) | \(x,y\) |

## Research Context

These architectures were developed to investigate different representations of the visual response of a photoelastic tactile sensor, including raw images, differential images, flattened intensity representations and learned intermediate physical quantities.

Together, the methods provide a comparison between direct image-to-output regression and a staged approach in which indentation depth and contact position are first estimated before force regression.

## Requirements

The individual methods use Python with libraries including PyTorch, torchvision, OpenCV, NumPy, pandas and scikit-learn.

Dependencies and setup instructions specific to each architecture are provided within its respective folder.

## Citation

If you use this repository or build upon this work in academic research, please cite the associated dissertation.

```bibtex
@mastersthesis{rajith2026photoelastic,
  author = {Staines Rajith},
  title  = {Photoelasticity-Based Tactile Sensing},
  year   = {2026}
}
```

## Licence

This repository is provided for research and educational use. See the repository licence for details.
