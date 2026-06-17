"""Evaluate a checkpoint on the GTSRB test split.

    python -m signsight.evaluate --checkpoint checkpoints/best.pt

Prints overall accuracy and a per-class report, and writes a confusion-matrix
image to outputs/confusion_matrix.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .classes import CLASS_NAMES
from .data import build_dataloaders
from .infer import load_model
from .train import pick_device


@torch.no_grad()
def collect_predictions(model, loader, device):
    y_true, y_pred = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        y_pred.extend(logits.argmax(1).cpu().tolist())
        y_true.extend(labels.tolist())
    return y_true, y_pred


def save_confusion_matrix(y_true, y_pred, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm, cmap="viridis")
    ax.set_title("GTSRB confusion matrix")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"confusion matrix -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on the GTSRB test set")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--report", action="store_true", help="print per-class precision/recall")
    args = parser.parse_args()

    device = pick_device()
    model = load_model(args.checkpoint, device)
    data = build_dataloaders(root=args.data_root, batch_size=args.batch_size, num_workers=2)

    y_true, y_pred = collect_predictions(model, data.test, device)
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    print(f"test accuracy: {acc*100:.2f}%  ({len(y_true)} images)")

    if args.report:
        from sklearn.metrics import classification_report

        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3))

    save_confusion_matrix(y_true, y_pred, Path("outputs/confusion_matrix.png"))


if __name__ == "__main__":
    main()
