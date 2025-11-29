import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import argparse
from PIL import Image

from src.models.evssm import EVSSM
from src.data.transforms import (
    CropToDivisibleBy32,
    CNN_NORMALIZATION_MEAN,
    CNN_NORMALIZATION_STD,
)
from src.utils.denormalize import denormalize
import albumentations as A
from albumentations.pytorch import ToTensorV2


def load_and_preprocess_image(image_path):
    """
    Load and preprocess a single image for model inference
    """
    # Load image using OpenCV (BGR format)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create transforms for single image (no dual target needed)
    transforms = A.Compose(
        [
            CropToDivisibleBy32(always_apply=True),
            A.Normalize(
                mean=CNN_NORMALIZATION_MEAN,
                std=CNN_NORMALIZATION_STD,
                always_apply=True,
            ),
            ToTensorV2(),
        ]
    )

    # Apply transforms
    transformed = transforms(image=image)
    processed_image = transformed["image"]

    # Add batch dimension
    processed_image = processed_image.unsqueeze(0)

    return processed_image, image


def save_results(original_image, deblurred_tensor, output_dir, image_name):
    """
    Save original and deblurred images
    """
    os.makedirs(output_dir, exist_ok=True)

    # Denormalize the output
    deblurred_denorm = denormalize(
        deblurred_tensor, CNN_NORMALIZATION_MEAN, CNN_NORMALIZATION_STD
    )
    deblurred_denorm = torch.clamp(deblurred_denorm, 0.0, 1.0)

    # Convert to numpy
    deblurred_np = deblurred_denorm[0].cpu().numpy().transpose(1, 2, 0)

    # Resize original image to match processed size for comparison
    original_resized = cv2.resize(
        original_image, (deblurred_np.shape[1], deblurred_np.shape[0])
    )

    # Save individual images
    base_name = os.path.splitext(image_name)[0]

    # Save original (resized)
    plt.figure(figsize=(10, 10))
    plt.imshow(original_resized)
    plt.axis("off")
    plt.title("Original Image", fontsize=16, fontweight="bold")
    plt.tight_layout(pad=0)
    plt.savefig(
        f"{output_dir}/{base_name}_original.jpg",
        format="jpg",
        dpi=150,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

    # Save deblurred
    plt.figure(figsize=(10, 10))
    plt.imshow(deblurred_np)
    plt.axis("off")
    plt.title("EVSSM Deblurred", fontsize=16, fontweight="bold")
    plt.tight_layout(pad=0)
    plt.savefig(
        f"{output_dir}/{base_name}_deblurred.jpg",
        format="jpg",
        dpi=150,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

    # Save side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    axes[0].imshow(original_resized)
    axes[0].set_title("Original Image", fontsize=18, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(deblurred_np)
    axes[1].set_title("EVSSM Deblurred", fontsize=18, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/{base_name}_comparison.jpg",
        format="jpg",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    return deblurred_np


def test_single_image(image_path, model_path, output_dir="src/results/single_test"):
    """
    Test a single image through the EVSSM model
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load and preprocess image
    print(f"Loading image from: {image_path}")
    try:
        processed_image, original_image = load_and_preprocess_image(image_path)
        print(f"Image loaded successfully. Processed shape: {processed_image.shape}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Initialize model
    print("Initializing EVSSM model...")
    model = EVSSM(
        inp_channels=3,
        out_channels=3,
        dim=16,
        num_blocks=[6, 6, 12],
        ffn_expansion_factor=3,
        bias=False,
    ).to(device)

    # Load trained weights
    if os.path.exists(model_path):
        print(f"Loading model weights from: {model_path}")
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
    else:
        print(f"Warning: Model weights not found at {model_path}")
        print("Using random weights!")

    # Run inference
    print("Running inference...")
    model.eval()
    with torch.no_grad():
        processed_image = processed_image.to(device)
        deblurred_output = model(processed_image)

    # Save results
    image_name = os.path.basename(image_path)
    print(f"Saving results to: {output_dir}")

    deblurred_np = save_results(
        original_image, deblurred_output, output_dir, image_name
    )

    print("✅ Processing complete!")
    print(f"Results saved in: {output_dir}")
    print(f"- Original: {os.path.splitext(image_name)[0]}_original.jpg")
    print(f"- Deblurred: {os.path.splitext(image_name)[0]}_deblurred.jpg")
    print(f"- Comparison: {os.path.splitext(image_name)[0]}_comparison.jpg")

    return deblurred_np


def main():
    parser = argparse.ArgumentParser(description="Test single image with EVSSM model")
    parser.add_argument(
        "--image", "-i", type=str, required=True, help="Path to input image (JPG/PNG)"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="/home/student/projects/Blur-removal/fftformer_model_184.pth",
        help="Path to model weights",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="src/results/single_test",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Check if image file exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return

    # Run test
    test_single_image(args.image, args.model, args.output)


if __name__ == "__main__":
    main()
