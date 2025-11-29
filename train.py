from pathlib import Path
from typing import Tuple
import contextlib

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import DataLoader

import mlflow
import numpy as np

from src.data.datasets import BlurImageDataset
from src.data.transforms import TEST_TRANSFORMS, TRAIN_TRANSFORMS
from src.utils.training import fit
from src.utils.validation import validate


def _prepare_dataloaders(cfg: DictConfig) -> Tuple[DataLoader, DataLoader]:
    blur_dir = to_absolute_path(cfg.data.blur_dir)
    sharp_dir = to_absolute_path(cfg.data.sharp_dir)

    # Пока для простоты используем одно и то же множество файлов,
    # разделение train/val реализуется через различные аугментации.
    train_dataset = BlurImageDataset(
        blur_dir=blur_dir, sharp_dir=sharp_dir, transforms=TRAIN_TRANSFORMS
    )
    val_dataset = BlurImageDataset(
        blur_dir=blur_dir, sharp_dir=sharp_dir, transforms=TEST_TRANSFORMS
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        drop_last=cfg.data.drop_last,
        pin_memory=cfg.data.pin_memory,
        num_workers=cfg.data.num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        pin_memory=cfg.data.pin_memory,
        num_workers=cfg.data.num_workers,
    )

    return train_loader, val_loader


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    # reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.train.device if torch.cuda.is_available() else "cpu")

    # data
    train_loader, val_loader = _prepare_dataloaders(cfg)

    # model
    model = hydra.utils.instantiate(cfg.model).to(device)

    # optimizer & scheduler через hydra.instantiate
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)

    save_dir = Path(to_absolute_path(cfg.train.save_dir))
    save_dir.mkdir(parents=True, exist_ok=True)

    # MLflow logging
    if cfg.logging.mlflow.enable:
        mlflow.set_tracking_uri(cfg.logging.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.logging.mlflow.experiment_name)

    with mlflow.start_run() if cfg.logging.mlflow.enable else contextlib.nullcontext():
        if cfg.logging.mlflow.enable:
            mlflow.log_params(
                {
                    "seed": cfg.seed,
                    "batch_size": cfg.data.batch_size,
                    "lr": cfg.optimizer.lr,
                    "epochs": cfg.train.epochs,
                    "model_name": cfg.model.name,
                }
            )

        for epoch in range(cfg.train.epochs):
            print(f"Epoch {epoch + 1} / {cfg.train.epochs}")

            train_loss, train_psnr, train_ssim = fit(
                model,
                train_loader,
                optimizer,
                device,
                epoch,
            )

            val_loss, val_psnr, val_ssim = validate(
                model,
                val_loader,
                device,
                epoch,
            )

            scheduler.step(val_loss)

            if cfg.logging.mlflow.enable:
                mlflow.log_metrics(
                    {
                        "train_loss": float(train_loss),
                        "val_loss": float(val_loss),
                        "train_psnr": float(train_psnr),
                        "val_psnr": float(val_psnr),
                        "train_ssim": float(train_ssim),
                        "val_ssim": float(val_ssim),
                    },
                    step=epoch,
                )

            ckpt_path = save_dir / f"{cfg.model.name}_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), ckpt_path)


if __name__ == "__main__":
    main()
