import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import glob
from tqdm import tqdm
import os

from src.data.datasets import BlurDataset
from src.data.transforms import (
    EVAL_TRANSFORMS,
    CNN_NORMALIZATION_MEAN,
    CNN_NORMALIZATION_STD,
)
from src.models.evssm import EVSSM
from src.utils.metrics import calculate_psnr, calculate_ssim
from src.utils.denormalize import denormalize
from src.utils.losses import FFTLoss
from torch.nn import L1Loss


def evaluate_model(model, dataloader, device, num_samples=100):
    """
    Evaluate model on the first num_samples from dataloader
    """
    model.eval()

    results = {
        "psnr_values": [],
        "ssim_values": [],
        "l1_losses": [],
        "fft_losses": [],
        "total_losses": [],
    }

    l1_loss = L1Loss()
    fft_loss = FFTLoss()
    lambda_fft = 0.1

    sample_count = 0

    with torch.no_grad():
        for batch_idx, (sharp_image, blur_image) in enumerate(
            tqdm(dataloader, desc="Evaluating")
        ):
            if sample_count >= num_samples:
                break

            blur_image = blur_image.to(device)
            sharp_image = sharp_image.to(device)

            # Get current batch size
            batch_size = blur_image.size(0)

            # Limit the batch to not exceed num_samples
            if sample_count + batch_size > num_samples:
                remaining = num_samples - sample_count
                blur_image = blur_image[:remaining]
                sharp_image = sharp_image[:remaining]
                batch_size = remaining

            outputs = model(blur_image)

            # Calculate losses
            loss_l1 = l1_loss(outputs, sharp_image)
            loss_fft = fft_loss(outputs, sharp_image)
            total_loss = loss_l1 + lambda_fft * loss_fft

            # Denormalize images
            outputs_denorm = denormalize(
                outputs, CNN_NORMALIZATION_MEAN, CNN_NORMALIZATION_STD
            )
            blur_denorm = denormalize(
                blur_image, CNN_NORMALIZATION_MEAN, CNN_NORMALIZATION_STD
            )
            sharp_image_denorm = denormalize(
                sharp_image, CNN_NORMALIZATION_MEAN, CNN_NORMALIZATION_STD
            )

            # Clamp values to [0,1] range
            outputs_denorm = torch.clamp(outputs_denorm, 0.0, 1.0)
            blur_denorm = torch.clamp(blur_denorm, 0.0, 1.0)
            sharp_image_denorm = torch.clamp(sharp_image_denorm, 0.0, 1.0)

            # Calculate metrics for each image in batch
            for i in range(batch_size):
                psnr_value = calculate_psnr(
                    outputs_denorm[i : i + 1], sharp_image_denorm[i : i + 1]
                )
                ssim_value = calculate_ssim(
                    outputs_denorm[i : i + 1], sharp_image_denorm[i : i + 1]
                )

                results["psnr_values"].append(psnr_value.item())
                results["ssim_values"].append(ssim_value.item())
                results["l1_losses"].append(loss_l1.item() / batch_size)
                results["fft_losses"].append(loss_fft.item() / batch_size)
                results["total_losses"].append(total_loss.item() / batch_size)

            sample_count += batch_size

            # Save some example images
            save_example_images(
                blur_denorm,
                outputs_denorm,
                sharp_image_denorm,
                batch_idx,
                model_name="EVSSM",
            )

    return results


def save_example_images(
    blur_images, deblurred_images, sharp_images, batch_idx, model_name="EVSSM"
):
    """
    Save example images for visualization in JPG format
    """
    base_dir = f"src/results/eval_{model_name.lower()}"
    os.makedirs(base_dir, exist_ok=True)

    # Create subdirectories for individual images
    os.makedirs(f"{base_dir}/blur", exist_ok=True)
    os.makedirs(f"{base_dir}/output", exist_ok=True)
    os.makedirs(f"{base_dir}/sharp", exist_ok=True)
    os.makedirs(f"{base_dir}/comparison", exist_ok=True)

    # Save first 4 images from the batch
    num_to_save = min(4, blur_images.size(0))

    for i in range(num_to_save):
        # Convert tensors to numpy and transpose for matplotlib
        blur_np = blur_images[i].cpu().numpy().transpose(1, 2, 0)
        deblurred_np = deblurred_images[i].cpu().numpy().transpose(1, 2, 0)
        sharp_np = sharp_images[i].cpu().numpy().transpose(1, 2, 0)

        # Save individual images in JPG format
        img_id = f"batch{batch_idx}_img{i}"

        # Save blur image
        plt.figure(figsize=(8, 8))
        plt.imshow(blur_np)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(
            f"{base_dir}/blur/{img_id}_blur.jpg",
            format="jpg",
            dpi=150,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()

        # Save model output
        plt.figure(figsize=(8, 8))
        plt.imshow(deblurred_np)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(
            f"{base_dir}/output/{img_id}_{model_name.lower()}_output.jpg",
            format="jpg",
            dpi=150,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()

        # Save sharp (ground truth) image
        plt.figure(figsize=(8, 8))
        plt.imshow(sharp_np)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(
            f"{base_dir}/sharp/{img_id}_sharp.jpg",
            format="jpg",
            dpi=150,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()

        # Save comparison image (3 images side by side)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(blur_np)
        axes[0].set_title("Blurred Input", fontsize=14, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(deblurred_np)
        axes[1].set_title(f"{model_name} Output", fontsize=14, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(sharp_np)
        axes[2].set_title("Ground Truth", fontsize=14, fontweight="bold")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(
            f"{base_dir}/comparison/{img_id}_comparison.jpg",
            format="jpg",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        # Also save a horizontal pair (blur vs output) for easy presentation
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        axes[0].imshow(blur_np)
        axes[0].set_title("Blurred Input", fontsize=16, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(deblurred_np)
        axes[1].set_title(f"{model_name} Output", fontsize=16, fontweight="bold")
        axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(
            f"{base_dir}/comparison/{img_id}_before_after.jpg",
            format="jpg",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()


def print_results(results, model_name="EVSSM"):
    """
    Print evaluation results
    """
    print(f"\n=== {model_name} Evaluation Results ===")
    print(f"Number of samples: {len(results['psnr_values'])}")
    print(
        f"Average PSNR: {np.mean(results['psnr_values']):.4f} ± {np.std(results['psnr_values']):.4f}"
    )
    print(
        f"Average SSIM: {np.mean(results['ssim_values']):.4f} ± {np.std(results['ssim_values']):.4f}"
    )
    print(
        f"Average L1 Loss: {np.mean(results['l1_losses']):.6f} ± {np.std(results['l1_losses']):.6f}"
    )
    print(
        f"Average FFT Loss: {np.mean(results['fft_losses']):.6f} ± {np.std(results['fft_losses']):.6f}"
    )
    print(
        f"Average Total Loss: {np.mean(results['total_losses']):.6f} ± {np.std(results['total_losses']):.6f}"
    )
    print("=" * 50)


def main():
    # Parameters
    random_seed = 42
    batch_size = 1  # Reduced batch size for evaluation
    num_samples = 100

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(random_seed)

    print(f"Using device: {device}")

    # Prepare data
    blur_list = np.array(sorted(glob.glob("src/datasets/numpy_arrays/blured/*.npy")))
    sharp_list = np.array(sorted(glob.glob("src/datasets/numpy_arrays/sharp/*.npy")))

    # Split into train and test sets (same as in train.py)
    indices = np.arange(len(blur_list))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx, test_idx = indices[:split], indices[split:]

    # Create test dataset WITHOUT TEST_TRANSFORMS to see original images
    test_dataset = BlurDataset(
        blur_list=blur_list[test_idx],
        sharp_list=sharp_list[test_idx],
        transforms=EVAL_TRANSFORMS,  # Using EVAL_TRANSFORMS instead of TEST_TRANSFORMS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep order for consistent evaluation
        pin_memory=True,
        num_workers=4,  # Reduced number of workers for evaluation
    )

    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Will evaluate first {num_samples} samples")

    # Initialize EVSSM model
    model = EVSSM(
        inp_channels=3,
        out_channels=3,
        dim=16,
        num_blocks=[6, 6, 12],
        ffn_expansion_factor=3,
        bias=False,
    ).to(device)

    # Load trained weights
    # You'll need to specify the path to your trained EVSSM model
    model_path = "/home/student/projects/Blur-removal/fftformer_model_184.pth"  # Adjust this path as needed

    if os.path.exists(model_path):
        print(f"Loading model weights from {model_path}")
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
    else:
        print(f"Warning: Model weights not found at {model_path}")
        print("Evaluating with random weights!")

    # Evaluate model
    results = evaluate_model(model, test_loader, device, num_samples)

    # Print results
    print_results(results, "EVSSM first")

    # Save results to file
    os.makedirs("src/results", exist_ok=True)
    np.save("src/results/evssm_first_eval_results.npy", results)


if __name__ == "__main__":
    main()
