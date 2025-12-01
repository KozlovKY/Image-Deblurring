from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


class PSNRMetric(nn.Module):
    """PSNR metric as a callable module."""

    def __init__(self, name: Optional[str] = None) -> None:
        """Initialize PSNR metric.

        Args:
            name: Optional name for the metric (ignored, kept for Hydra compatibility).
        """
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred, target, reduction="mean")
        if mse == 0:
            return torch.tensor(float("inf"), device=pred.device)
        max_pixel = 1.0
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr


class SSIMMetric(nn.Module):
    """SSIM metric as a callable module."""

    def __init__(self, name: Optional[str] = None) -> None:
        """Initialize SSIM metric.

        Args:
            name: Optional name for the metric (ignored, kept for Hydra compatibility).
        """
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ssim_value = ssim(pred, target, data_range=1.0, size_average=True)
        return ssim_value
