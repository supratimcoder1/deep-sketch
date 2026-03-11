"""PatchGAN (70×70) Discriminator for Pix2Pix."""

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


class _DiscriminatorBlock(nn.Module):
    """Conv → InstanceNorm → LeakyReLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=4, stride=stride, padding=1, bias=False,
            ),
        ]
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PatchGANDiscriminator(nn.Module):
    """70×70 PatchGAN discriminator.

    Input : concatenated photo and sketch ``(B, 6, 256, 256)``
    Output: patch prediction map  ``(B, 1, 30, 30)``
    """

    def __init__(self, in_channels: int = 6) -> None:
        super().__init__()

        self.model = nn.Sequential(
            # 256 → 128  (no norm on first layer)
            _DiscriminatorBlock(in_channels, 64, stride=2, use_norm=False),
            # 128 → 64
            _DiscriminatorBlock(64, 128, stride=2),
            # 64 → 32
            _DiscriminatorBlock(128, 256, stride=2),
            # 32 → 31  (stride=1)
            _DiscriminatorBlock(256, 512, stride=1),
            # 31 → 30  (final 1-channel prediction)
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
        )

        self.apply(_init_weights)

    def forward(self, photo: torch.Tensor, sketch: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            photo:  ``(B, 3, 256, 256)``
            sketch: ``(B, 3, 256, 256)``

        Returns:
            Patch prediction map ``(B, 1, H', W')``.
        """
        x = torch.cat([photo, sketch], dim=1)  # (B, 6, 256, 256)
        return self.model(x)
