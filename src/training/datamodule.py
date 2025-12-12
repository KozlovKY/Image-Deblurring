from typing import Optional

import lightning.pytorch as pl
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data.datasets import BlurImageDataset
from src.data.transforms import build_transforms


class DeblurDataModule(pl.LightningDataModule):
    """Lightning DataModule for the blur/deblur dataset."""

    def __init__(
        self,
        train_blur_dir: str,
        train_sharp_dir: str,
        val_blur_dir: str,
        val_sharp_dir: str,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        drop_last: bool,
        transforms_cfg: DictConfig,
    ) -> None:
        super().__init__()
        self.train_blur_dir = train_blur_dir
        self.train_sharp_dir = train_sharp_dir
        self.val_blur_dir = val_blur_dir
        self.val_sharp_dir = val_sharp_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        self.train_dataset: Optional[BlurImageDataset] = None
        self.val_dataset: Optional[BlurImageDataset] = None

        self.train_transforms = build_transforms(transforms_cfg, split="train")
        self.val_transforms = build_transforms(transforms_cfg, split="val")

    def setup(self, stage: Optional[str] = None) -> None:
        train_blur_dir = to_absolute_path(self.train_blur_dir)
        train_sharp_dir = to_absolute_path(self.train_sharp_dir)
        val_blur_dir = to_absolute_path(self.val_blur_dir)
        val_sharp_dir = to_absolute_path(self.val_sharp_dir)

        self.train_dataset = BlurImageDataset(
            blur_dir=train_blur_dir,
            sharp_dir=train_sharp_dir,
            transforms=self.train_transforms,
        )
        self.val_dataset = BlurImageDataset(
            blur_dir=val_blur_dir,
            sharp_dir=val_sharp_dir,
            transforms=self.val_transforms,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            self.setup("fit")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            self.setup("fit")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )
