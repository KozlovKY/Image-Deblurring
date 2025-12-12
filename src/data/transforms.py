"""Image augmentation and preprocessing pipelines with Hydra config support."""

import albumentations as A
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig


class CropToDivisibleByFactor(A.DualTransform):
    """Crop image to make both height and width divisible by `factor`."""

    def __init__(
        self,
        factor: int,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super().__init__(always_apply, p)
        self.factor = factor

    def apply(self, img, **params):
        h, w = img.shape[:2]
        new_h = (h // self.factor) * self.factor
        new_w = (w // self.factor) * self.factor

        start_y = (h - new_h) // 2
        start_x = (w - new_w) // 2

        return img[start_y : start_y + new_h, start_x : start_x + new_w]

    def apply_to_bbox(self, bbox, **params):
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint

    def get_transform_init_args_names(self):
        return ("factor",)


def build_transforms(cfg: DictConfig, split: str) -> A.Compose:
    """
    Build augmentation pipeline from Hydra config.
    Returns:
        Configured albumentations.Compose pipeline
    """

    norm_transform = A.Normalize(
        mean=cfg.cnn_normalization_mean,
        std=cfg.cnn_normalization_std,
        always_apply=True,
    )

    if split == "train":
        return A.Compose(
            [
                A.RandomCrop(
                    width=cfg.crop_size, height=cfg.crop_size, always_apply=True
                ),
                A.HorizontalFlip(p=cfg.flip_prob),
                A.VerticalFlip(p=cfg.flip_prob),
                A.RandomRotate90(p=cfg.random_rotate90_prob),
                A.OneOf(
                    [
                        A.Affine(
                            scale=tuple(cfg.affine_scale_range),
                            rotate=tuple(cfg.affine_rotate_range),
                            shear=tuple(cfg.affine_shear_range),
                            p=1.0,
                        ),
                        A.NoOp(p=1.0),
                    ],
                    p=cfg.affine_or_noop_prob,
                ),
                norm_transform,
                ToTensorV2(),
            ],
            additional_targets={"image0": "image"},
        )

    elif split == "test":
        return A.Compose(
            [
                A.CenterCrop(
                    width=cfg.crop_size, height=cfg.crop_size, always_apply=True
                ),
                norm_transform,
                ToTensorV2(),
            ],
            additional_targets={"image0": "image"},
        )

    elif split == "val":
        return A.Compose(
            [
                CropToDivisibleByFactor(
                    factor=cfg.downsample_factor,
                    always_apply=True,
                ),
                norm_transform,
                ToTensorV2(),
            ],
            additional_targets={"image0": "image"},
        )

    else:
        raise ValueError(f"Unknown split: {split}. Use 'train', 'test', or 'val'")
