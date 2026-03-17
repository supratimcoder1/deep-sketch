"""CUFS paired photo-sketch dataset loader with Albumentations augmentations."""

import os
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def get_paired_spatial_transforms(image_size: int = 256) -> A.Compose:
    """Return spatial augmentations applied identically to both photo and sketch."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(-0.05, 0.05),
                rotate=(-5, 5),
                p=0.5,
            ),
            A.RandomResizedCrop(
                size=(image_size, image_size), # Fixed: uses 'size' tuple instead of height/width
                scale=(0.9, 1.0),
                p=1.0,
            ),
        ],
        additional_targets={"sketch": "image"},
    )


def get_photo_only_transforms() -> A.Compose:
    """Return photometric augmentations applied only to the photo."""
    return A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        ],
    )


def get_train_transforms(image_size: int = 256) -> dict[str, A.Compose]:
    """Return training augmentation pipelines (paired spatial + photo-only).

    Returns a dict with keys ``"spatial"`` and ``"photo_only"``.
    """
    return {
        "spatial": get_paired_spatial_transforms(image_size),
        "photo_only": get_photo_only_transforms(),
    }


def get_val_transforms(image_size: int = 256) -> A.Compose:
    """Return validation/inference transforms (resize only, no augmentation)."""
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
        ],
        additional_targets={"sketch": "image"},
    )


def _to_tensor_normalized(image: np.ndarray) -> torch.Tensor:
    """Convert a HWC uint8 image to a CHW float32 tensor normalised to [-1, 1]."""
    image = image.astype(np.float32) / 255.0  # [0, 1]
    image = (image * 2.0) - 1.0                # [-1, 1]
    image = np.transpose(image, (2, 0, 1))     # HWC -> CHW
    return torch.from_numpy(image)


class CUFSDataset(Dataset):
    """PyTorch Dataset for the CUFS paired photo-sketch dataset.

    Each sample is a dict with keys ``"photo"`` and ``"sketch"``, both
    tensors of shape ``(3, 256, 256)`` normalised to ``[-1, 1]``.
    """

    def __init__(
        self,
        photo_dir: str,
        sketch_dir: str,
        transform: dict[str, A.Compose] | A.Compose | None = None,
        image_size: int = 256,
    ) -> None:
        super().__init__()
        self.photo_dir = photo_dir
        self.sketch_dir = sketch_dir
        self.image_size = image_size

        # transform can be a dict (train) or a single Compose (val) or None
        if isinstance(transform, dict):
            self.spatial_transform: A.Compose | None = transform.get("spatial")
            self.photo_only_transform: A.Compose | None = transform.get("photo_only")
        elif isinstance(transform, A.Compose):
            self.spatial_transform = transform
            self.photo_only_transform = None
        else:
            self.spatial_transform = None
            self.photo_only_transform = None

        # Sorted filenames ensure deterministic pairing
        self.filenames: list[str] = sorted(
            f
            for f in os.listdir(photo_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        fname = self.filenames[index]

        photo = cv2.imread(os.path.join(self.photo_dir, fname))
        sketch = cv2.imread(os.path.join(self.sketch_dir, fname))

        # OpenCV loads as BGR – convert to RGB
        photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
        sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2RGB)

        # 1) Paired spatial augmentations (applied identically to both)
        if self.spatial_transform is not None:
            augmented: dict[str, Any] = self.spatial_transform(
                image=photo, sketch=sketch
            )
            photo = augmented["image"]
            sketch = augmented["sketch"]
        else:
            photo = cv2.resize(photo, (self.image_size, self.image_size))
            sketch = cv2.resize(sketch, (self.image_size, self.image_size))

        # 2) Photo-only photometric augmentations (NOT applied to sketch)
        if self.photo_only_transform is not None:
            photo = self.photo_only_transform(image=photo)["image"]

        photo_tensor = _to_tensor_normalized(photo)
        sketch_tensor = _to_tensor_normalized(sketch)

        return {"photo": photo_tensor, "sketch": sketch_tensor}
