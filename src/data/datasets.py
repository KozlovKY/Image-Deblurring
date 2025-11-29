from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import torch.utils.data as data


class BlurImageDataset(data.Dataset):
    """Датасет для размытия / восстановления по исходным PNG/JPEG изображениям.

    Ожидается, что в двух директориях лежат файлы с одинаковыми именами
    (или совпадающими по сортировке), в одной — размытые картинки, в другой —
    соответствующие им резкие.
    """

    def __init__(
        self,
        blur_dir: str,
        sharp_dir: str,
        transforms: Optional[Callable] = None,
        image_exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg"),
    ) -> None:
        super().__init__()
        blur_dir_path = Path(blur_dir)
        sharp_dir_path = Path(sharp_dir)

        if not blur_dir_path.is_dir() or not sharp_dir_path.is_dir():
            raise ValueError(
                f"Blur or sharp directory does not exist: {blur_dir}, {sharp_dir}"
            )

        blur_files = sorted(
            [p for p in blur_dir_path.rglob("*") if p.suffix.lower() in image_exts]
        )
        sharp_files = sorted(
            [p for p in sharp_dir_path.rglob("*") if p.suffix.lower() in image_exts]
        )

        if len(blur_files) == 0 or len(sharp_files) == 0:
            raise ValueError(
                f"No image files found in blur_dir={blur_dir} or sharp_dir={sharp_dir}"
            )

        if len(blur_files) != len(sharp_files):
            raise ValueError(
                f"Mismatch between number of blur and sharp images: "
                f"{len(blur_files)} vs {len(sharp_files)}"
            )

        self.blur_files = blur_files
        self.sharp_files = sharp_files
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.blur_files)

    def __getitem__(self, idx: int):
        blur_path = str(self.blur_files[idx])
        sharp_path = str(self.sharp_files[idx])

        blur = cv2.imread(blur_path, cv2.IMREAD_COLOR)
        sharp = cv2.imread(sharp_path, cv2.IMREAD_COLOR)

        if blur is None or sharp is None:
            raise RuntimeError(f"Failed to read images: {blur_path} or {sharp_path}")

        # BGR -> RGB
        blur = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
        sharp = cv2.cvtColor(sharp, cv2.COLOR_BGR2RGB)

        data = {"image": blur, "image0": sharp}

        if self.transforms:
            augmented = self.transforms(**data)
            blur = augmented["image"]
            sharp = augmented["image0"]

        return sharp, blur
