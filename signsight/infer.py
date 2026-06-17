"""Load a checkpoint and predict on a single image.

    python -m signsight.infer --checkpoint checkpoints/best.pt --image sign.png

Also exposes load_model() / predict() so the FastAPI server reuses the exact
same preprocessing and decoding path as the CLI.
"""

from __future__ import annotations

import argparse
from typing import NamedTuple

import torch
from PIL import Image

from .classes import CLASS_NAMES
from .data import build_transforms
from .model import build_model

_EVAL_TF = build_transforms(train=False)


class Prediction(NamedTuple):
    class_id: int
    label: str
    confidence: float


def load_model(checkpoint_path: str, device: str | torch.device = "cpu") -> torch.nn.Module:
    """Rebuild the architecture recorded in the checkpoint and load weights."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_model(ckpt.get("model", "cnn"), ckpt.get("num_classes", len(CLASS_NAMES)))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model


@torch.no_grad()
def predict(model: torch.nn.Module, image: Image.Image, top_k: int = 3,
            device: str | torch.device = "cpu") -> list[Prediction]:
    """Return the top-k predictions for a PIL image."""
    x = _EVAL_TF(image.convert("RGB")).unsqueeze(0).to(device)
    probs = torch.softmax(model(x), dim=1)[0]
    top = torch.topk(probs, k=min(top_k, probs.numel()))
    return [
        Prediction(int(i), CLASS_NAMES[int(i)], float(p))
        for p, i in zip(top.values, top.indices)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a traffic sign from an image")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    preds = predict(model, Image.open(args.image), top_k=args.top_k)
    for rank, p in enumerate(preds, 1):
        print(f"{rank}. {p.label:40s} {p.confidence*100:5.1f}%  (class {p.class_id})")


if __name__ == "__main__":
    main()
