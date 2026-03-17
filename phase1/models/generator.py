"""Pix2Pix U-Net Generator with InstanceNorm and skip connections."""

from typing import Callable

import torch
import torch.nn as nn


def _init_weights(module: nn.Module) -> None:
    """Pix2Pix weight initialisation convention."""
    classname = module.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(module.weight.data, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias.data)
    elif "InstanceNorm" in classname:
        if module.weight is not None:
            nn.init.normal_(module.weight.data, mean=1.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias.data)


class _EncoderBlock(nn.Module):
    """Encoder block: Conv → InstanceNorm → LeakyReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
        ]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _DecoderBlock(nn.Module):
    """Decoder block: ConvTranspose → InstanceNorm → (Dropout) → ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetGenerator(nn.Module):
    """U-Net Generator for Pix2Pix.

    Input  shape: ``(B, 3, 256, 256)``
    Output shape: ``(B, 3, 256, 256)``  (Tanh activation → values in [-1, 1])
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3) -> None:
        super().__init__()

        # ---- Encoder ----
        # 256 → 128
        self.enc1 = _EncoderBlock(in_channels, 64, use_norm=False)
        # 128 → 64
        self.enc2 = _EncoderBlock(64, 128)
        # 64 → 32
        self.enc3 = _EncoderBlock(128, 256)
        # 32 → 16
        self.enc4 = _EncoderBlock(256, 512)
        # 16 → 8
        self.enc5 = _EncoderBlock(512, 512)
        # 8 → 4
        self.enc6 = _EncoderBlock(512, 512)
        # 4 → 2
        self.enc7 = _EncoderBlock(512, 512)
        # 2 → 1  (bottleneck)
        self.enc8 = _EncoderBlock(512, 512, use_norm=False)

        # ---- Decoder (with skip connections) ----
        # 1 → 2
        self.dec1 = _DecoderBlock(512, 512, use_dropout=True)
        # 2 → 4
        self.dec2 = _DecoderBlock(1024, 512, use_dropout=True)
        # 4 → 8
        self.dec3 = _DecoderBlock(1024, 512, use_dropout=True)
        # 8 → 16
        self.dec4 = _DecoderBlock(1024, 512)
        # 16 → 32
        self.dec5 = _DecoderBlock(1024, 256)
        # 32 → 64
        self.dec6 = _DecoderBlock(512, 128)
        # 64 → 128
        self.dec7 = _DecoderBlock(256, 64)

        # 128 → 256
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)   # (B, 64, 128, 128)
        e2 = self.enc2(e1)  # (B, 128, 64, 64)
        e3 = self.enc3(e2)  # (B, 256, 32, 32)
        e4 = self.enc4(e3)  # (B, 512, 16, 16)
        e5 = self.enc5(e4)  # (B, 512, 8, 8)
        e6 = self.enc6(e5)  # (B, 512, 4, 4)
        e7 = self.enc7(e6)  # (B, 512, 2, 2)
        e8 = self.enc8(e7)  # (B, 512, 1, 1)  bottleneck

        # Decoder with skip connections (cat along channel dim)
        d1 = self.dec1(e8)                          # (B, 512, 2, 2)
        d2 = self.dec2(torch.cat([d1, e7], dim=1))  # (B, 512, 4, 4)
        d3 = self.dec3(torch.cat([d2, e6], dim=1))  # (B, 512, 8, 8)
        d4 = self.dec4(torch.cat([d3, e5], dim=1))  # (B, 512, 16, 16)
        d5 = self.dec5(torch.cat([d4, e4], dim=1))  # (B, 256, 32, 32)
        d6 = self.dec6(torch.cat([d5, e3], dim=1))  # (B, 128, 64, 64)
        d7 = self.dec7(torch.cat([d6, e2], dim=1))  # (B, 64, 128, 128)

        out = self.final(torch.cat([d7, e1], dim=1))  # (B, 3, 256, 256)
        return out
