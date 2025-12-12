from typing import Any, Dict, Optional

import hydra
import lightning.pytorch as pl
import torch
from omegaconf import DictConfig

from src.utils.denormalize import denormalize


class LitDeblurring(pl.LightningModule):
    """LightningModule, wrapping the deblurring model and loss/metrics of the project."""

    def __init__(
        self,
        net: Any,
        optimizer: Any,
        scheduler: Optional[Any] = None,
        loss_cfg: Optional[DictConfig] = None,
        metrics_cfg: Optional[DictConfig] = None,
    ) -> None:
        super().__init__()
        # Handle net - can be DictConfig or already instantiated
        if isinstance(net, DictConfig):
            self.net = hydra.utils.instantiate(net)
        else:
            self.net = net

        # Handle optimizer - can be DictConfig or already instantiated (partial)
        if isinstance(optimizer, DictConfig):
            self.optimizer = hydra.utils.instantiate(optimizer)
        else:
            self.optimizer = optimizer

        # Handle scheduler - can be DictConfig or already instantiated (partial)
        if scheduler is not None:
            if isinstance(scheduler, DictConfig):
                self.scheduler = hydra.utils.instantiate(scheduler)
            else:
                self.scheduler = scheduler
        else:
            self.scheduler = None

        # Single loss
        loss_config = loss_cfg["components"][0]
        self.loss_fn = hydra.utils.instantiate(loss_config)

        # Metrics
        self.metrics = {}
        for metric_item in metrics_cfg["components"]:
            metric_name = metric_item["name"]
            metric_fn = hydra.utils.instantiate(metric_item)
            self.metrics[metric_name] = metric_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _denormalize_for_metrics(
        self, outputs: torch.Tensor, sharp_image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Denormalize and clamp images to [0, 1] range for metric computation."""
        outputs_denorm = denormalize(outputs)
        sharp_denorm = denormalize(sharp_image)
        outputs_denorm = torch.clamp(outputs_denorm, 0.0, 1.0)
        sharp_denorm = torch.clamp(sharp_denorm, 0.0, 1.0)
        return outputs_denorm, sharp_denorm

    def _shared_step(self, batch: Any, stage: str) -> Dict[str, torch.Tensor]:
        sharp_image, blur_image = batch
        blur_image = blur_image.to(self.device)
        sharp_image = sharp_image.to(self.device)

        outputs = self(blur_image)

        # compute loss
        total_loss = self.loss_fn(outputs, sharp_image)

        # denormalize for metrics
        outputs_denorm, sharp_denorm = self._denormalize_for_metrics(
            outputs, sharp_image
        )

        # log loss
        self.log(
            f"{stage}_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True
        )

        # compute and log metrics from config
        for metric_name, metric_fn in self.metrics.items():
            metric_val = metric_fn(outputs_denorm, sharp_denorm)
            self.log(
                f"{stage}_{metric_name}",
                metric_val,
                on_step=False,
                on_epoch=True,
                prog_bar=(stage == "val"),
            )

        return {"loss": total_loss}

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        result = self._shared_step(batch, stage="train")
        return result["loss"]

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self._shared_step(batch, stage="val")

    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.parameters())

        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
