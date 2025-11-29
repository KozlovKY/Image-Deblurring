import os
import sys

# Опционально: добавляем корневую директорию проекта в sys.path
# Если запускать приложение из корневой директории, это может быть не нужно
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import torch
import torch.nn as nn
import numpy as np

from data.transforms import (
    EVAL_TRANSFORMS,
    CNN_NORMALIZATION_MEAN,
    CNN_NORMALIZATION_STD,
)
from utils.denormalize import denormalize

from torchvision import transforms
from PIL import Image


def deblur_image(
    input_image_path: str, output_image_path: str, model: nn.Module, device: str
) -> None:
    """
    Загружает изображение по пути input_image_path, применяет к нему модель
    для восстановления резкости, и сохраняет результат по пути output_image_path.
    """

    image = Image.open(input_image_path).convert("RGB")
    np_img = np.array(image)
    transformed_image = EVAL_TRANSFORMS(image=np_img)["image"].to(device)
    with torch.no_grad():
        result_image = model(transformed_image[None, ...])
    result_image_denorm = denormalize(
        result_image, CNN_NORMALIZATION_MEAN, CNN_NORMALIZATION_STD
    )

    # Обрезаем значения до диапазона [0,1]
    result_image_denorm = torch.clamp(result_image_denorm, 0.0, 1.0)

    output_image = transforms.ToPILImage()(result_image_denorm.squeeze(0))
    output_image.save(output_image_path)
