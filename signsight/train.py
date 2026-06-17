"""Config-driven training loop.

    python -m signsight.train --config configs/default.yaml

Tracks validation accuracy each epoch and saves the best checkpoint (weights +
the config + class count) so evaluate.py / infer.py / the server can reload it
without guessing the architecture.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from .data import NUM_CLASSES, build_dataloaders
from .model import build_model, count_parameters


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False, desc="train" if train else "val"):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += images.size(0)
    return loss_sum / total, correct / total


def train(config: dict) -> Path:
    device = pick_device()
    print(f"device: {device}")

    data = build_dataloaders(
        root=config["data_root"],
        batch_size=config["batch_size"],
        val_fraction=config["val_fraction"],
        num_workers=config["num_workers"],
        seed=config["seed"],
    )

    model = build_model(config["model"], NUM_CLASSES).to(device)
    print(f"model: {config['model']}  ({count_parameters(model):,} params)")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = None
    if config.get("scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    ckpt_dir = Path(config["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"
    best_acc = 0.0

    for epoch in range(1, config["epochs"] + 1):
        tr_loss, tr_acc = run_epoch(model, data.train, criterion, optimizer, device, train=True)
        va_loss, va_acc = run_epoch(model, data.val, criterion, optimizer, device, train=False)
        if scheduler:
            scheduler.step()
        print(
            f"epoch {epoch:2d}/{config['epochs']}  "
            f"train {tr_loss:.3f}/{tr_acc*100:.1f}%  val {va_loss:.3f}/{va_acc*100:.1f}%"
        )
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(
                {"state_dict": model.state_dict(), "model": config["model"],
                 "num_classes": NUM_CLASSES, "val_acc": va_acc},
                best_path,
            )
            print(f"  saved best -> {best_path}  (val acc {va_acc*100:.2f}%)")

    print(f"done. best val acc: {best_acc*100:.2f}%")
    return best_path


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the signsight model")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, help="override epochs from config")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    os.makedirs(config["data_root"], exist_ok=True)
    train(config)


if __name__ == "__main__":
    main()
