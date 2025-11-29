import torch
from tqdm import tqdm

from .metrics import calculate_psnr, calculate_ssim
from .denormalize import denormalize
from ..data.transforms import CNN_NORMALIZATION_MEAN, CNN_NORMALIZATION_STD
from .losses import FFTLoss
from torch.nn import L1Loss


def validate(model, dataloader, device, epoch):
    model.eval()
    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    num_batches = len(dataloader)

    l1_loss = L1Loss()
    fft_loss = FFTLoss()
    lambda_fft = 0.1

    with torch.no_grad():
        for i, (sharp_image, blur_image) in enumerate(tqdm(dataloader)):
            blur_image = blur_image.to(device)
            sharp_image = sharp_image.to(device)
            outputs = model(blur_image)
            loss_l1 = l1_loss(outputs, sharp_image)
            loss_fft = fft_loss(outputs, sharp_image)
            loss = loss_l1 + lambda_fft * loss_fft

            running_loss += loss.item()

            # Денормализация изображений
            outputs_denorm = denormalize(
                outputs, CNN_NORMALIZATION_MEAN, CNN_NORMALIZATION_STD
            )
            blur_denorm = denormalize(
                blur_image, CNN_NORMALIZATION_MEAN, CNN_NORMALIZATION_STD
            )
            sharp_image_denorm = denormalize(
                sharp_image, CNN_NORMALIZATION_MEAN, CNN_NORMALIZATION_STD
            )

            # Обрезаем значения до диапазона [0,1]
            outputs_denorm = torch.clamp(outputs_denorm, 0.0, 1.0)
            blur_denorm = torch.clamp(blur_denorm, 0.0, 1.0)
            sharp_image_denorm = torch.clamp(sharp_image_denorm, 0.0, 1.0)

            # Вычисляем PSNR и SSIM
            psnr_value = calculate_psnr(outputs_denorm, sharp_image_denorm)
            ssim_value = calculate_ssim(outputs_denorm, sharp_image_denorm)

            running_psnr += psnr_value.item()
            running_ssim += ssim_value.item()

    val_loss = running_loss / num_batches
    val_psnr = running_psnr / num_batches
    val_ssim = running_ssim / num_batches

    return val_loss, val_psnr, val_ssim
