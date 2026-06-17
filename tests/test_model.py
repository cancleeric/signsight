"""Model unit tests -- shapes and gradient flow, no dataset needed."""

import pytest
import torch

from signsight.data import IMAGE_SIZE, NUM_CLASSES
from signsight.model import TrafficSignCNN, build_model, count_parameters


def test_forward_shape():
    model = build_model("cnn")
    x = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)
    out = model(x)
    assert out.shape == (4, NUM_CLASSES)


def test_has_params_and_gradcam_layer():
    model = TrafficSignCNN()
    assert count_parameters(model) > 0
    assert isinstance(model.gradcam_layer, torch.nn.Conv2d)


def test_backward_flows():
    model = build_model("cnn")
    out = model(torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_model("not-a-model")
