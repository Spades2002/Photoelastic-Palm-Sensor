"""
Physics-informed model architecture.

Both depth and contact location are learned CNN predictions, never known
inputs, since neither is available at deployment: pressing the sensor
doesn't tell the computer where or how hard you pressed, that's exactly
what the network has to figure out from the images.

DepthCNN predicts depth_mm from the fringe delta image.
LocalisationCNN predicts (x_mm, y_mm) from the indent delta image.
Both predictions are concatenated and passed through an MLP that regresses
Fx, Fy, Fz and the combined force magnitude. depth_mm and (x_mm, y_mm) from
the CSV are used only as training labels for their respective CNNs, never
as inputs anywhere in the forward pass.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class SingleImageCNN(nn.Module):
    """ResNet-18 backbone adapted for single-channel input, regressing
    out_dim scalar targets from one image."""

    def __init__(self, out_dim: int, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Adapt the first conv layer to accept 1-channel grayscale input,
        # averaging the pretrained RGB filters into a single channel so the
        # pretrained low-level features are still usable.
        original_conv1 = backbone.conv1
        new_conv1 = nn.Conv2d(
            1,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False,
        )
        if pretrained:
            with torch.no_grad():
                new_conv1.weight = nn.Parameter(original_conv1.weight.mean(dim=1, keepdim=True))
        backbone.conv1 = new_conv1

        backbone.fc = nn.Linear(backbone.fc.in_features, out_dim)
        self.backbone = backbone

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.backbone(image)  # (B, out_dim)


class DepthCNN(SingleImageCNN):
    """Predicts a single scalar, indentation depth in mm, from the fringe
    delta image."""

    def __init__(self, pretrained: bool = True):
        super().__init__(out_dim=1, pretrained=pretrained)

    def forward(self, fringe_image: torch.Tensor) -> torch.Tensor:
        return super().forward(fringe_image).squeeze(-1)  # (B,)


class LocalisationCNN(SingleImageCNN):
    """Predicts (x_mm, y_mm) contact location from the indent delta image."""

    def __init__(self, pretrained: bool = True):
        super().__init__(out_dim=2, pretrained=pretrained)

    def forward(self, indent_image: torch.Tensor) -> torch.Tensor:
        return super().forward(indent_image)  # (B, 2)


class ForceMLP(nn.Module):
    """Regresses [Fx, Fy, Fz, magnitude] from [depth_mm, x_mm, y_mm]."""

    def __init__(self, hidden_dims: tuple = (64, 64), dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = 3
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 4))
        self.net = nn.Sequential(*layers)

    def forward(self, depth_mm: torch.Tensor, xy_mm: torch.Tensor) -> torch.Tensor:
        features = torch.cat([depth_mm.unsqueeze(-1), xy_mm], dim=-1)
        return self.net(features)


class TactileForceNet(nn.Module):
    """Full pipeline: a depth CNN and a localisation CNN both predict from
    images alone, feeding an MLP force regressor. Nothing in forward() ever
    receives a ground-truth value, everything downstream of the two raw
    images is a learned prediction."""

    def __init__(self, pretrained_backbone: bool = True, mlp_hidden_dims: tuple = (64, 64)):
        super().__init__()
        self.depth_cnn = DepthCNN(pretrained=pretrained_backbone)
        self.localisation_cnn = LocalisationCNN(pretrained=pretrained_backbone)
        self.force_mlp = ForceMLP(hidden_dims=mlp_hidden_dims)

    def forward(self, fringe_image: torch.Tensor, indent_image: torch.Tensor) -> dict:
        depth_pred = self.depth_cnn(fringe_image)
        xy_pred = self.localisation_cnn(indent_image)
        force_pred = self.force_mlp(depth_pred, xy_pred)
        return {"depth_pred": depth_pred, "xy_pred": xy_pred, "force_pred": force_pred}
