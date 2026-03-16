"""Pix2Pix loss functions with label smoothing."""

import torch
import torch.nn as nn


def generator_loss(
    fake_pred: torch.Tensor,
    fake_img: torch.Tensor,
    real_img: torch.Tensor,
    lambda_l1: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute generator loss.

    Loss = BCE(fake_pred, 1) + lambda * L1(fake, real)

    Args:
        fake_pred: Discriminator prediction on the generated image.
        fake_img:  Generator output.
        real_img:  Ground-truth sketch.
        lambda_l1: Weight for L1 reconstruction loss.

    Returns:
        Tuple of (total_loss, gan_loss, l1_loss).
    """
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    gan_loss = bce(fake_pred, torch.ones_like(fake_pred))
    l1_loss = l1(fake_img, real_img)
    total_loss = gan_loss + lambda_l1 * l1_loss

    return total_loss, gan_loss, l1_loss


def discriminator_loss(
    real_pred: torch.Tensor,
    fake_pred: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute discriminator loss with label smoothing.

    Real labels are set to 0.9 for training stability.

    Args:
        real_pred: Discriminator output on real pairs.
        fake_pred: Discriminator output on fake pairs.

    Returns:
        Tuple of (total_loss, real_loss, fake_loss).
    """
    bce = nn.BCEWithLogitsLoss()

    real_loss = bce(real_pred, torch.full_like(real_pred, 0.9))
    fake_loss = bce(fake_pred, torch.zeros_like(fake_pred))
    total_loss = (real_loss + fake_loss) * 0.5

    return total_loss, real_loss, fake_loss
