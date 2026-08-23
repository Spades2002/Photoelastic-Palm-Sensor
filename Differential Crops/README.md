# Photoelastic Tactile Sensor: Differential-Crop Force and Contact Localisation

Deep learning pipeline for estimating **normal contact force** and **contact location** from differential camera images captured by a photoelastic tactile sensor.

The model uses a dual-branch architecture based on pretrained **ResNet-18** networks. Rather than operating directly on raw camera crops, the network receives differential images obtained by subtracting an unloaded reference frame from the corresponding loaded sensor image.

The network predicts:

- Normal contact force, \(F_z\)
- Contact position, \((x,y)\)

The localisation prediction is additionally conditioned on the estimated normal force.

## Model Architecture

The network retains the two-branch ResNet-18 structure used for the raw-image architecture but replaces the raw inputs with their corresponding differential crops.

```text
Fringe delta crop ──> ResNet-18 ──┐
                                  │
                                  ├──> Concatenated features ──> Force head ──> Fz
                                  │                                  │
Mirror delta crop ──> ResNet-18 ──┘                                  │
                                  │                                  │
                                  └────────────── + predicted Fz <───┘
                                                     │
                                                     └──> Localisation head
                                                              │
                                                              └──> x, y
```

Each ResNet-18 produces a 512-dimensional feature embedding. The two embeddings are concatenated to form a joint 1024-dimensional visual representation.

The force head operates on this representation to estimate the normal contact force:

```text
Fz_pred = force_head(concatenated_features)
```

The predicted normal force is then appended to the visual representation before it is supplied to the localisation head:

```text
location_input = [concatenated_features ; Fz_pred]
location_pred  = location_head(location_input)
```

Importantly, the force prediction is **detached** before being supplied to the localisation head. The localisation loss therefore cannot propagate gradients back through the force prediction.

This allows the localisation network to use the estimated normal force as an additional conditioning variable while keeping force estimation independent of the localisation loss during backpropagation.

## Differential Image Representation

The network operates on differential rather than raw image crops.

For a loaded image \(I\) and its corresponding unloaded reference image \(I_0\), the differential representation is formed from:

```text
ΔI = I - I₀
```

In the implementation, the signed difference is shifted to a mid-grey value of 128 before being stored as an 8-bit image:

```python
diff = current_crop.astype(np.int16) - baseline_crop.astype(np.int16)
delta_crop = np.clip(diff + 128, 0, 255).astype(np.uint8)
```

As a result:

```text
128       -> no change from the unloaded state
> 128     -> pixel became brighter under load
< 128     -> pixel became darker under load
```

The sign of the intensity change is therefore retained rather than using an absolute difference.

Each loaded image is compared against the corresponding unloaded `0 mm` baseline associated with the same measurement location, cycle and phase.

## Network Inputs

Two differential regions of interest are supplied to the network.

### Fringe Delta Crop

The fringe delta crop represents the change in the photoelastic fringe pattern relative to the unloaded sensor state.

```text
Raw fringe crop
       │
       ├── subtract corresponding unloaded fringe crop
       │
       └──> Fringe delta crop ──> ResNet-18
```

This branch provides the visual representation used for estimating changes associated with the applied normal force.

### Mirror Delta Crop

The mirror delta crop represents the change in the reflected indenter-contact region relative to its unloaded state.

```text
Raw mirror crop
       │
       ├── subtract corresponding unloaded mirror crop
       │
       └──> Mirror delta crop ──> ResNet-18
```

The resulting visual features contribute to the contact-localisation prediction.

Both crops are resized to `224 × 224` pixels and normalised using ImageNet statistics before being passed to their respective ResNet-18 backbones.

## Force-Conditioned Localisation

Unlike the raw-crop architecture, the localisation head is explicitly conditioned on the predicted normal force.

The model first computes:

```text
fringe_embedding = ResNet18(fringe_delta)
mirror_embedding = ResNet18(mirror_delta)

features = concat(fringe_embedding, mirror_embedding)
```

Normal force is then estimated:

```text
Fz_pred = force_head(features)
```

For localisation, the force prediction is detached from the computational graph:

```text
location_input = concat(features, Fz_pred.detach())
location_pred = location_head(location_input)
```

The resulting information flow is therefore:

```text
Differential image features ────────> Fz
             │                        │
             │                        │ detached
             │                        ▼
             └──────────────────────> x, y
```

This means \(F_z\) can influence the localisation prediction during the forward pass, but the localisation loss cannot modify the force predictor through this connection.

## Multi-Task Objective

The force and localisation outputs are trained jointly using an equally weighted multi-task objective.

The force term is applied only to the normal force \(F_z\), while the localisation term is applied to the two-dimensional contact position \((x,y)\):

```text
L = MSE(Fz_pred, Fz_true)
  + MSE(location_pred, location_true)
```

where:

```text
Force target:         Fz
Localisation targets: x_mm, y_mm
```

Both loss terms have equal weighting.

The architecture therefore jointly optimises normal-force estimation and contact localisation while maintaining the one-way conditioning relationship from force estimation to localisation.

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
| `config.py` | Defines dataset paths, output directories, baseline depth and ROI configuration. |
| `dataset_index.py` | Links force measurements with image folders and experimental metadata. |
| `feature_extraction.py` | Handles image loading, ROI extraction and image-processing operations. |
| `build_dataset.py` | Constructs the training dataset and generates the differential image crops. |
| `train_resnet.py` | Contains the dual-ResNet model, dataset class and training pipeline. |
| `train.py` | Entry point configured specifically for the differential force-conditioned architecture. |
| `evaluate_resnet.py` | Evaluates the saved checkpoint on held-out contact locations. |
| `predict.py` | Performs inference using a new loaded image and its corresponding unloaded baseline. |
| `requirements.txt` | Python dependencies required by the pipeline. |

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

Install the required dependencies:

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

The unloaded reference depth is configured as:

```python
BASELINE_DEPTH_MM = 0.0
```

These unloaded frames are used to construct the differential crops.

The ground-truth targets used by this architecture are:

```text
Fz
x_mm
y_mm
```

Although the source dataset may contain additional force components, this architecture trains the force head specifically against the normal-force component \(F_z\).

## Building the Differential Dataset

Generate the dataset and differential crops using:

```bash
python build_dataset.py --save-crops --phases down
```

Differential crops are the default crop type.

For every loaded sensor image, the pipeline identifies the corresponding unloaded baseline and generates:

```text
fringe_delta_path
indent_delta_path
```

These paths are stored in the generated `dataset.csv` and subsequently loaded during model training.

## Training

Train the differential-crop model using:

```bash
python train.py --epochs 30 --batch-size 32
```

The training entry point locks the architecture to:

```python
crop_source="delta"
force_targets=["Fz"]
force_conditions_location=True
```

This ensures that training corresponds specifically to the differential-crop, force-conditioned architecture.

The two ResNet-18 backbones are initialised using pretrained ImageNet weights and trained jointly with the regression heads.

Training uses a spatially grouped train-validation split based on the physical contact point. Images belonging to a held-out contact location therefore do not appear in the training set.

The best-performing model checkpoint is saved to the configured models directory.

## Evaluation

Evaluate the trained model using:

```bash
python evaluate_resnet.py
```

The evaluation script loads the saved checkpoint and reconstructs its architecture from the stored configuration.

Performance is evaluated for:

```text
Normal force:         Fz
Contact localisation: x_mm, y_mm
```

The evaluation uses the same spatially held-out validation split defined during training.

## Inference

Because this architecture operates on differential images, inference requires both a loaded sensor image and its corresponding unloaded reference image.

Run:

```bash
python predict.py \
    --image path/to/loaded_photo.png \
    --baseline-image path/to/0mm_photo.png \
    --model resnet
```

The pipeline extracts the fringe and mirror regions from both images, constructs the differential crops, applies the required preprocessing and passes them through the trained network.

Example output:

```text
{
    "Fz": ...,
    "x_mm": ...,
    "y_mm": ...
}
```

## Method Summary

The complete inference pipeline can be summarised as:

```text
                 Loaded camera image
                         │
             ┌───────────┴───────────┐
             │                       │
        Fringe ROI              Mirror ROI
             │                       │
       subtract I₀              subtract I₀
             │                       │
             ▼                       ▼
     Fringe delta crop        Mirror delta crop
             │                       │
         ResNet-18                ResNet-18
             │                       │
             └───────────┬───────────┘
                         │
                Concatenated features
                         │
                 ┌───────┴────────┐
                 │                │
                 ▼                │
             Force head           │
                 │                │
                 ▼                │
                Fz                │
                 │                │
              detach             │
                 │                │
                 └───────┬────────┘
                         │
                         ▼
                 Localisation head
                         │
                         ▼
                       x, y
```

The architecture therefore combines differential photoelastic information with force-conditioned localisation to jointly estimate normal contact force and two-dimensional contact position.

## Citation

If you use this repository or build upon this work in academic research, please cite the associated publication or dissertation.

```bibtex
@misc{photoelastic_tactile_sensor_differential,
  title  = {Differential Photoelastic Tactile Sensing for Force Estimation and Contact Localisation},
  author = {Staines Rajith},
  year   = {2026}
}
```

## Licence

This repository is provided for research and educational use. Add an appropriate licence file before redistribution or external reuse.