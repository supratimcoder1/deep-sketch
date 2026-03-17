"""Reuse Phase 1 Pix2Pix loss functions for Phase 2."""

from phase1.training.loss_functions import discriminator_loss, generator_loss


__all__ = ["generator_loss", "discriminator_loss"]