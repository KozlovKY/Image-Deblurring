from __future__ import annotations

from typing import Any, Dict, Optional

import hydra
import lightning.pytorch as pl
import torch
from omegaconf import DictConfig
from torch import nn

from src.utils.denormalize import denormalize
from src.utils.losses import FFTLoss
from src.utils.metrics import calculate_psnr, calculate_ssim
from src.data.transforms import CNN_NORMALIZATION_MEAN, CNN_NORMALIZATION_STD


class LitDeblurring(pl.LightningModule):
    """LightningModule, оборачивающий модель деблюринга и лосс/метрики проекта."""

    def __init__(
        self,
        net: nn.Module,
        optimizer_cfg: DictConfig,
        scheduler_cfg: Optional[DictConfig] = None,
        model_name: str = "model",
    ) -> None:
        super().__init__()
        self.net = net
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.model_name = model_name

        self.l1_loss = nn.L1Loss()
        self.fft_loss = FFTLoss()
        self.lambda_fft = 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _shared_step(self, batch: Any, stage: str) -> Dict[str, torch.Tensor]:
        sharp_image, blur_image = batch
        blur_image = blur_image.to(self.device)
        sharp_image = sharp_image.to(self.device)

        outputs = self(blur_image)

        loss_l1 = self.l1_loss(outputs, sharp_image)
        loss_fft = self.fft_loss(outputs, sharp_image)
        loss = loss_l1 + self.lambda_fft * loss_fft

        # Денормализация и метрики
        outputs_denorm = denormalize(
            outputs, CNN_NORMALIZATION_MEAN, CNN_NORMALIZATION_STD
        )
        sharp_denorm = denormalize(
            sharp_image, CNN_NORMALIZATION_MEAN, CNN_NORMALIZATION_STD
        )

        outputs_denorm = torch.clamp(outputs_denorm, 0.0, 1.0)
        sharp_denorm = torch.clamp(sharp_denorm, 0.0, 1.0)

        psnr = calculate_psnr(outputs_denorm, sharp_denorm)
        ssim = calculate_ssim(outputs_denorm, sharp_denorm)

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            f"{stage}_psnr",
            psnr,
            on_step=False,
            on_epoch=True,
            prog_bar=(stage == "val"),
        )
        self.log(
            f"{stage}_ssim",
            ssim,
            on_step=False,
            on_epoch=True,
            prog_bar=(stage == "val"),
        )

        return {"loss": loss}

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        result = self._shared_step(batch, stage="train")
        return result["loss"]

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self._shared_step(batch, stage="val")

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.optimizer_cfg, params=self.net.parameters()
        )
        if self.scheduler_cfg is None:
            return optimizer

        scheduler = hydra.utils.instantiate(self.scheduler_cfg, optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
