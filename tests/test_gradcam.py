"""Grad-CAM produces a normalised heatmap matching the input resolution."""

import numpy as np
import torch

from signsight.data import IMAGE_SIZE
from signsight.gradcam import GradCAM
from signsight.model import build_model


def test_gradcam_shape_and_range():
    model = build_model("cnn")
    cam = GradCAM(model, model.gradcam_layer)(torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert cam.shape == (IMAGE_SIZE, IMAGE_SIZE)
    assert cam.dtype == np.float32 or cam.dtype == np.float64
    assert 0.0 <= cam.min() and cam.max() <= 1.0 + 1e-6
