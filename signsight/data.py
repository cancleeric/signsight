"""GTSRB dataset loading, augmentation, and dataloaders.

GTSRB = German Traffic Sign Recognition Benchmark: 43 classes of traffic
signs, ~39k training and ~12k test images of varying size. torchvision can
download it for us, so this module only has to define the transforms and wrap
everything in dataloaders.

Design choices worth noting:
  * Signs are upright and direction-sensitive (a "turn left" must not become a
    "turn right"), so augmentation uses small rotations / affine jitter only --
    never a horizontal flip.
  * Everything is resized to a fixed 32x32 so the model input is constant.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import GTSRB

NUM_CLASSES = 43
IMAGE_SIZE = 32

# Channel statistics for GTSRB (close enough; the model is robust to small
# deviations and BatchNorm absorbs the rest).
MEAN = (0.3403, 0.3121, 0.3214)
STD = (0.2724, 0.2608, 0.2669)


def build_transforms(train: bool) -> transforms.Compose:
    """Augmentation for training; plain resize+normalise for eval."""
    if train:
        return transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomRotation(12),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


@dataclass
class DataBundle:
    train: DataLoader
    val: DataLoader
    test: DataLoader


def build_dataloaders(
    root: str = "data",
    batch_size: int = 128,
    val_fraction: float = 0.1,
    num_workers: int = 2,
    seed: int = 42,
    download: bool = True,
) -> DataBundle:
    """Return train/val/test dataloaders, carving a val split out of train."""
    train_full = GTSRB(root=root, split="train", download=download,
                       transform=build_transforms(train=True))
    test_set = GTSRB(root=root, split="test", download=download,
                     transform=build_transforms(train=False))

    n_val = int(len(train_full) * val_fraction)
    n_train = len(train_full) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(train_full, [n_train, n_val], generator=generator)

    # The val split should not be augmented; swap in the eval transform.
    val_set.dataset = GTSRB(root=root, split="train", download=False,
                            transform=build_transforms(train=False))

    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return DataBundle(
        train=DataLoader(train_set, shuffle=True, **common),
        val=DataLoader(val_set, shuffle=False, **common),
        test=DataLoader(test_set, shuffle=False, **common),
    )
