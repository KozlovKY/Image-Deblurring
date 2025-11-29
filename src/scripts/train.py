import sys

# sys.path.append("..") # Removed
# sys.path.append("../utils") # Removed
# sys.path.append("../data") # Removed
# sys.path.append("../models") # Removed

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data.datasets import BlurDataset
from src.data.transforms import TRAIN_TRANSFORMS, TEST_TRANSFORMS
from src.models.deblurnet import DeblurCNN
from src.models.unet import UNet
from src.models.attention_unet import AttentionUNet
from src.models.attention_fourie_uner import fftformer  # Added fftformer import
from src.models.evssm import EVSSM  # Added fftformer import
from src.utils.training import fit
from src.utils.validation import validate
from src.utils.losses import FFTLoss  # Added FFTLoss import
from torch.nn import L1Loss  # Added L1Loss import

import numpy as np
import glob
import wandb


def main():
    # Гиперпараметры
    random_seed = 42
    batch_size = 16
    lr = 5e-4
    perceptual_weight = 0.1
    epochs = 150

    # Инициализация wandb
    wandb.init(
        # id='oylo24kk',
        project="deblurring_project",
        name="fftformer_run",  # Changed wandb run name
        # resume='allow',
        config={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "criterion": "l1+0.1*fft",  # Updated criterion name in config
            # 'perceptual_weight': perceptual_weight
        },
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(random_seed)

    # Подготовка данных
    blur_list = np.array(sorted(glob.glob("src/datasets/numpy_arrays/blured/*.npy")))
    sharp_list = np.array(sorted(glob.glob("src/datasets/numpy_arrays/sharp/*.npy")))

    # Разделение на обучающую и тестовую выборки
    indices = np.arange(len(blur_list))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx, test_idx = indices[:split], indices[split:]

    train_dataset = BlurDataset(
        blur_list=blur_list[train_idx],
        sharp_list=sharp_list[train_idx],
        transforms=TRAIN_TRANSFORMS,
    )

    test_dataset = BlurDataset(
        blur_list=blur_list[test_idx],
        sharp_list=sharp_list[test_idx],
        transforms=TEST_TRANSFORMS,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=8,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=8,
    )

    # Инициализация модели
    # model = DeblurCNN().to(device)
    # model = UNet().to(device)
    # model = AttentionUNet().to(device)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = DeblurringResUNet().to(device) # Commented out
    model = EVSSM(
        inp_channels=3,
        out_channels=3,
        dim=16,
        num_blocks=[6, 6, 12],
        num_refinement_blocks=4,
        ffn_expansion_factor=3,
        bias=False,
    ).to(
        device
    )  # Added fftformer instantiation
    # best_model = wandb.restore('unet_attn_combined_model_137.pth', run_path="tiltovskii/deblurring_project/oylo24kk")
    # model.load_state_dict(torch.load('unet_attn_combined_model_119.pth', weights_only=True))

    # criterion = torch.nn.MSELoss() # Removed criterion creation
    # criterion = CombinedLoss(perceptual_weight) # Removed criterion creation
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    wandb.watch(model, log="all")

    for epoch in range(0, epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_psnr, train_epoch_ssim = fit(
            model, train_loader, optimizer, device, epoch  # Removed criterion argument
        )

        val_epoch_loss, val_epoch_psnr, val_epoch_ssim = validate(
            model, test_loader, device, epoch  # Removed criterion argument
        )
        scheduler.step(val_epoch_loss)

        # Логирование средних метрик за эпоху
        wandb.log(
            {
                "Avg Train Loss": train_epoch_loss,
                "Avg Val Loss": val_epoch_loss,
                "Avg Train PSNR": train_epoch_psnr,
                "Avg Val PSNR": val_epoch_psnr,
                "Avg Train RSSIM": train_epoch_ssim,
                "Avg Val SSIM": val_epoch_ssim,
                "epoch": epoch,
            }
        )

        # Сохранение модели
        torch.save(
            model.state_dict(), f"fftformer_model_{epoch + 1}.pth"
        )  # Changed saved model name
        wandb.save(f"fftformer_model_{epoch + 1}.pth")  # Changed saved model name

    print("Training complete.")
    wandb.finish()


if __name__ == "__main__":
    main()
