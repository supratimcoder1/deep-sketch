"""Phase 2 Pix2Pix loss functions with optional VGG perceptual regularization."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VGGLoss(nn.Module):
    """Perceptual L1 loss on VGG16 feature maps."""

    def __init__(self, device: torch.device):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:23]
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.criterion = nn.L1Loss()

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1),
        )

        # Keep all submodules/buffers initialized on the requested device.
        self.to(device)

    def forward(self, fake_image: torch.Tensor, real_image: torch.Tensor) -> torch.Tensor:
        # Runtime safety: if caller moves inputs to a different device, follow once here.
        input_device = fake_image.device
        if self.mean.device != input_device:
            self.to(input_device)

        # Keep perceptual path in fp32 for numerical stability with AMP training.
        mean = self.mean.to(device=input_device, dtype=torch.float32)
        std = self.std.to(device=input_device, dtype=torch.float32)
        fake_fp32 = fake_image.float()
        real_fp32 = real_image.float()

        with torch.autocast(device_type=input_device.type, enabled=False):
            fake_norm = ((fake_fp32 + 1.0) / 2.0 - mean) / std
            real_norm = ((real_fp32 + 1.0) / 2.0 - mean) / std
            fake_features = self.vgg(fake_norm)
            real_features = self.vgg(real_norm)

        return self.criterion(fake_features, real_features)


def discriminator_loss(
    real_pred: torch.Tensor,
    fake_pred: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute discriminator loss and return total + components as tensors."""
    criterion = nn.BCEWithLogitsLoss()
    real_target = torch.ones_like(real_pred)
    fake_target = torch.zeros_like(fake_pred)

    loss_real = criterion(real_pred, real_target)
    loss_fake = criterion(fake_pred, fake_target)
    d_loss = (loss_real + loss_fake) / 2.0

    return d_loss, loss_real, loss_fake


def edge_loss(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """Edge-aware L1 loss using Sobel gradients on grayscale projections."""
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3).to(fake.device)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3).to(fake.device)

    def get_edges(x: torch.Tensor) -> torch.Tensor:
        gray = torch.mean(x.float(), dim=1, keepdim=True)
        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        # epsilon avoids non-finite gradients when gx=gy=0 at many pixels.
        return torch.sqrt(gx**2 + gy**2 + 1e-8)

    return F.l1_loss(get_edges(fake), get_edges(real))


def generator_loss(
    fake_pred: torch.Tensor,
    fake_targets: torch.Tensor,
    real_targets: torch.Tensor,
    vgg_criterion: VGGLoss | None = None,
    lambda_l1: float = 90.0,
    lambda_vgg: float = 8.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute generator loss and return total + key components as tensors.

    Return contract is always a 3-tuple to keep call sites safe when unpacking.
    """
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()

    real_target = torch.ones_like(fake_pred)

    gan_loss = criterion_gan(fake_pred, real_target)
    l1_loss = criterion_l1(fake_targets, real_targets)

    g_loss = gan_loss + (lambda_l1 * l1_loss)

    if vgg_criterion is not None:
        vgg_loss = vgg_criterion(fake_targets, real_targets)
        g_loss += (lambda_vgg * vgg_loss)

    lambda_edge = 3.0
    edge = edge_loss(fake_targets, real_targets)
    g_loss += lambda_edge * edge

    return g_loss, gan_loss, l1_loss
