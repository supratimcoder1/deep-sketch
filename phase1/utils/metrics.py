"""Evaluation metrics: LPIPS and SSIM."""

import torch
import lpips
from torchmetrics.image import StructuralSimilarityIndexMeasure


class PerceptualMetrics:
    """Wrapper around LPIPS and SSIM for evaluation.

    Both metrics expect 3-channel images normalised to ``[-1, 1]``.
    """

    def __init__(self, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        self.lpips_fn = lpips.LPIPS(net="alex").to(self.device)
        self.lpips_fn.eval()
        self.ssim_fn = StructuralSimilarityIndexMeasure(data_range=2.0).to(self.device)

    @torch.no_grad()
    def compute_lpips(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> float:
        """Mean LPIPS over a batch (lower is better)."""
        pred = pred.to(self.device)
        target = target.to(self.device)
        score = self.lpips_fn(pred, target)  # (B, 1, 1, 1)
        return score.mean().item()

    @torch.no_grad()
    def compute_ssim(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> float:
        """Mean SSIM over a batch (higher is better)."""
        pred = pred.to(self.device)
        target = target.to(self.device)
        return self.ssim_fn(pred, target).item()
