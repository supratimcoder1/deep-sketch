"""Sketch-to-stylized-color paired dataset loader for Phase 2."""

from __future__ import annotations

import os
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def get_paired_spatial_transforms(image_size: int = 256) -> A.Compose:
    """Return shared spatial transforms for sketch and stylized target."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.97, 1.03),
                translate_percent=(-0.03, 0.03),
                rotate=(-4, 4),
                p=0.4,
            ),
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.92, 1.0),
                ratio=(0.97, 1.03),
                p=1.0,
            ),
        ],
        additional_targets={"target": "image"},
    )


def get_train_transforms(image_size: int = 256) -> A.Compose:
    """Return the training augmentation pipeline."""
    return get_paired_spatial_transforms(image_size)


def get_val_transforms(image_size: int = 256) -> A.Compose:
    """Return validation/inference transforms."""
    return A.Compose(
        [A.Resize(height=image_size, width=image_size)],
        additional_targets={"target": "image"},
    )


def _to_tensor_normalized(image: np.ndarray) -> torch.Tensor:
    image = image.astype(np.float32) / 255.0
    image = (image * 2.0) - 1.0
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image)


class SketchToColorDataset(Dataset):
    """Paired dataset mapping line sketches to stylized color portraits."""

    def __init__(
        self,
        sketch_dir: str,
        target_dir: str,
        transform: A.Compose | None = None,
        image_size: int = 256,
    ) -> None:
        super().__init__()
        self.sketch_dir = sketch_dir
        self.target_dir = target_dir
        self.transform = transform
        self.image_size = image_size
        self.filenames = sorted(
            name
            for name in os.listdir(sketch_dir)
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
            and os.path.exists(os.path.join(target_dir, name))
        )

        if not self.filenames:
            raise FileNotFoundError(
                f"No paired files found between {sketch_dir} and {target_dir}"
            )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        filename = self.filenames[index]
        sketch = cv2.imread(os.path.join(self.sketch_dir, filename), cv2.IMREAD_COLOR)
        target = cv2.imread(os.path.join(self.target_dir, filename), cv2.IMREAD_COLOR)
        if sketch is None or target is None:
            raise FileNotFoundError(f"Failed to read paired images for {filename}")

        sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented: dict[str, Any] = self.transform(image=sketch, target=target)
            sketch = augmented["image"]
            target = augmented["target"]
        else:
            sketch = cv2.resize(sketch, (self.image_size, self.image_size))
            target = cv2.resize(target, (self.image_size, self.image_size))

        return {
            "sketch": _to_tensor_normalized(sketch),
            "target": _to_tensor_normalized(target),
        }