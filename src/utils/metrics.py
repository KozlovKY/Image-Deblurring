import torch
import torch.nn.functional as F
from math import log10
from pytorch_msssim import ssim


def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2, reduction="mean")
    if mse == 0:
        return float("inf")
    max_pixel = 1.0  # Предполагаем, что изображения нормализованы в диапазоне [0,1]
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr


def calculate_ssim(img1, img2):
    # Используем pytorch_msssim для вычисления SSIM
    ssim_value = ssim(img1, img2, data_range=1.0, size_average=True)
    return ssim_value
