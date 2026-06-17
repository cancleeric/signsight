"""Grad-CAM: visualise which pixels drove a prediction.

Grad-CAM (Selvaraju et al., 2017) weights the feature maps of the last
convolutional layer by the gradient of the target class flowing into them,
giving a coarse heatmap of "where the network looked". It's the standard
sanity check that a CV model learned the sign and not the background.

    python -m signsight.gradcam --checkpoint checkpoints/best.pt --image sign.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data import IMAGE_SIZE, build_transforms
from .infer import load_model


class GradCAM:
    """Hook the target conv layer, capture activations + gradients, combine."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model.eval()
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inp, out):
        self.activations = out.detach()

    def _save_gradient(self, _module, _grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, class_id: int | None = None) -> np.ndarray:
        logits = self.model(x)
        if class_id is None:
            class_id = int(logits.argmax(1))
        self.model.zero_grad()
        logits[0, class_id].backward()

        # Global-average-pool the gradients to get per-channel weights, then
        # take a ReLU'd weighted sum of the activation maps.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)      # (1, C, 1, 1)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE),
                            mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def overlay(image: Image.Image, cam: np.ndarray) -> Image.Image:
    """Blend the heatmap over the (resized) input image."""
    import matplotlib.cm as mcm

    base = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    heat = (mcm.jet(cam)[:, :, :3] * 255).astype(np.uint8)
    blended = (0.5 * np.asarray(base) + 0.5 * heat).astype(np.uint8)
    return Image.fromarray(blended)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grad-CAM for a single image")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="outputs/gradcam.png")
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    if not hasattr(model, "gradcam_layer"):
        raise SystemExit("this checkpoint's model has no gradcam_layer (use the 'cnn' model)")

    image = Image.open(args.image)
    x = build_transforms(train=False)(image.convert("RGB")).unsqueeze(0)
    cam = GradCAM(model, model.gradcam_layer)(x)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay(image, cam).save(out_path)
    print(f"saved Grad-CAM overlay -> {out_path}")


if __name__ == "__main__":
    main()
