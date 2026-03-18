"""Phase 2 Pix2Pix loss functions with VGG Perceptual Loss for structural sharpness."""

import torch
import torch.nn as nn
import torchvision.models as models

class VGGLoss(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        # Load pre-trained VGG16 and extract the first few feature-rich layers
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16].to(device)
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.criterion = nn.L1Loss()
        
        # ImageNet normalization constants required for VGG
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

    def forward(self, fake_image: torch.Tensor, real_image: torch.Tensor) -> torch.Tensor:
        # Denormalize from [-1, 1] to [0, 1] then apply ImageNet normalization
        fake_norm = ((fake_image + 1) / 2.0 - self.mean) / self.std
        real_norm = ((real_image + 1) / 2.0 - self.mean) / self.std
        
        fake_features = self.vgg(fake_norm)
        real_features = self.vgg(real_norm)
        
        return self.criterion(fake_features, real_features)

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
    vgg_criterion: VGGLoss | None = None,
    lambda_l1: float = 100.0,
    lambda_vgg: float = 10.0
) -> tuple[torch.Tensor, float, float]:
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()
    
    real_target = torch.ones_like(fake_pred)
    
    gan_loss = criterion_gan(fake_pred, real_target)
    l1_loss = criterion_l1(fake_targets, real_targets)
    
    g_loss = gan_loss + (lambda_l1 * l1_loss)
    
    if vgg_criterion is not None:
        vgg_loss = vgg_criterion(fake_targets, real_targets)
        g_loss += (lambda_vgg * vgg_loss)
    
    return g_loss, gan_loss.item(), l1_loss.item()