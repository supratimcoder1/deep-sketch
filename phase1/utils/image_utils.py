"""Image utility helpers: denormalize, convert, and save grids."""

from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Convert tensor from [-1, 1] to [0, 1]."""
    return (tensor + 1.0) / 2.0


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a CHW tensor in [-1, 1] to a HWC uint8 numpy array."""
    tensor = denormalize(tensor).clamp(0.0, 1.0)
    # CHW → HWC
    image = tensor.detach().cpu().permute(1, 2, 0).numpy()
    return (image * 255.0).astype(np.uint8)


def save_image_grid(
    photos: torch.Tensor,
    real_sketches: torch.Tensor,
    fake_sketches: torch.Tensor,
    save_path: str | Path,
    max_images: int = 4,
) -> None:
    """Save a side-by-side grid: photo | real sketch | generated sketch.

    Args:
        photos:        Batch tensor ``(B, 3, H, W)`` in ``[-1, 1]``.
        real_sketches: Batch tensor ``(B, 3, H, W)`` in ``[-1, 1]``.
        fake_sketches: Batch tensor ``(B, 3, H, W)`` in ``[-1, 1]``.
        save_path:     Destination file path.
        max_images:    Maximum number of rows to display.
    """
    n = min(max_images, photos.size(0))
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))

    if n == 1:
        axes = axes[np.newaxis, :]  # ensure 2-D indexing

    for i in range(n):
        axes[i, 0].imshow(tensor_to_image(photos[i]))
        axes[i, 0].set_title("Photo")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(tensor_to_image(real_sketches[i]))
        axes[i, 1].set_title("Real Sketch")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(tensor_to_image(fake_sketches[i]))
        axes[i, 2].set_title("Generated Sketch")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
