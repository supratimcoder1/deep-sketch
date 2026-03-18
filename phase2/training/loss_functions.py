"""Phase 2 Pix2Pix loss functions with tuned L1 weight for stylized sharpness."""

import torch
import torch.nn as nn

def discriminator_loss(real_pred: torch.Tensor, fake_pred: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    criterion = nn.BCEWithLogitsLoss()
    real_target = torch.ones_like(real_pred)
    fake_target = torch.zeros_like(fake_pred)
    
    loss_real = criterion(real_pred, real_target)
    loss_fake = criterion(fake_pred, fake_target)
    d_loss = (loss_real + loss_fake) / 2.0
    
    return d_loss, loss_real.item(), loss_fake.item()

def generator_loss(
    fake_pred: torch.Tensor, 
    fake_targets: torch.Tensor, 
    real_targets: torch.Tensor, 
    lambda_l1: float = 30.0  # Reduced from 100 to stop color averaging and force sharp polygon edges
) -> tuple[torch.Tensor, float, float]:
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()
    
    real_target = torch.ones_like(fake_pred)
    
    gan_loss = criterion_gan(fake_pred, real_target)
    l1_loss = criterion_l1(fake_targets, real_targets)
    
    g_loss = gan_loss + (lambda_l1 * l1_loss)
    
    return g_loss, gan_loss.item(), l1_loss.item()