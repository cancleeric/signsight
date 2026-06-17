"""Model definitions.

Two options, selectable from config:
  * ``cnn``      -- a compact 3-block convolutional net written from scratch.
                    Small enough to train on CPU, strong enough for ~98% on
                    GTSRB. This is the default and the one Grad-CAM targets.
  * ``resnet18`` -- torchvision ResNet-18 with the final layer swapped for 43
                    classes, to show transfer learning.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .data import NUM_CLASSES


def _conv_block(c_in: int, c_out: int) -> nn.Sequential:
    """Conv -> BatchNorm -> ReLU -> MaxPool, halving the spatial size."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class TrafficSignCNN(nn.Module):
    """A small 3-block CNN for 32x32 traffic-sign images.

    32x32 -> 16x16 (32ch) -> 8x8 (64ch) -> 4x4 (128ch) -> FC -> logits.
    The final conv block (``features[2]``) is the natural Grad-CAM target.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    @property
    def gradcam_layer(self) -> nn.Module:
        """The last convolution, used as the Grad-CAM activation hook."""
        return self.features[2][0]


def build_model(name: str = "cnn", num_classes: int = NUM_CLASSES) -> nn.Module:
    """Factory used by training/serving so the choice lives in config."""
    name = name.lower()
    if name == "cnn":
        return TrafficSignCNN(num_classes)
    if name == "resnet18":
        from torchvision import models

        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        return net
    raise ValueError(f"unknown model '{name}' (expected 'cnn' or 'resnet18')")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
