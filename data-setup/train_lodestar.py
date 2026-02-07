# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "numpy",
#     "pillow",
#     "torch",
#     "deeptrack",
#     "deeplay",
#     "matplotlib",
#     "pint",
# ]
# ///

"""
Train a LodeSTAR model for particle tracking on microscopy images.

Usage:
    uv run train_lodestar.py --input "Au Cit Trial 2 LI 20%" --output lodestar_model.pth
"""

import argparse
import os
import logging
import warnings

# Suppress pint logging
logging.getLogger("pint").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
from PIL import Image
import torch
import deeptrack as dt
import deeplay as dl
import matplotlib.pyplot as plt


def load_and_preprocess_images(input_path, max_images=None):
    """Load images from directory and normalize them."""
    if os.path.isdir(input_path):
        files = sorted(
            [
                os.path.join(input_path, f)
                for f in os.listdir(input_path)
                if f.lower().endswith((".png", ".jpg"))
            ]
        )
    else:
        raise ValueError("Input must be a directory containing images")

    if not files:
        raise ValueError(f"No PNG or JPG files found in {input_path}")

    if max_images:
        files = files[:max_images]

    print(f"Loading {len(files)} images...")

    # Load images as grayscale and stack
    images = []
    for f in files:
        img = Image.open(f).convert("L")
        img_array = np.array(img).astype(np.float32)
        images.append(img_array)

    data = np.stack(images)

    # Normalize to [0, 1]
    data = (data - np.min(data)) / (np.ptp(data) + 1e-8)

    print(f"Loaded data shape: {data.shape}")
    return data


def create_training_crops(data, crop_size=128, num_crops=5):
    """
    Create training crops from the loaded images.

    Args:
        data: Array of shape (num_images, height, width)
        crop_size: Size of square crop
        num_crops: Number of random crops to extract

    Returns:
        List of crops
    """
    num_images, height, width = data.shape
    crops = []

    for _ in range(num_crops):
        # Random image
        img_idx = np.random.randint(0, num_images)
        img = data[img_idx]

        # Random crop location
        if height >= crop_size and width >= crop_size:
            y = np.random.randint(0, height - crop_size + 1)
            x = np.random.randint(0, width - crop_size + 1)
            crop = img[y : y + crop_size, x : x + crop_size]
        else:
            # If image is smaller than crop size, use the whole image
            crop = img

        crops.append(crop)

    return crops


def visualize_sample(data, num_samples=3):
    """Visualize sample images from the dataset."""
    fig, axes = plt.subplots(1, min(num_samples, len(data)), figsize=(15, 5))
    if num_samples == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        if i < len(data):
            ax.imshow(data[i], cmap="gray")
            ax.set_title(f"Image {i + 1}")
            ax.axis("off")

    plt.tight_layout()
    plt.savefig("training_samples.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Sample images saved to training_samples.png")


def train_lodestar(
    input_path,
    output_path,
    crop_size=128,
    num_crops=10,
    n_transforms=4,
    max_epochs=50,
    batch_size=4,
    learning_rate=1e-3,
    use_gpu=True,
    visualize=True,
):
    """
    Train a LodeSTAR model on particle tracking images.

    Args:
        input_path: Directory containing training images
        output_path: Path to save trained model
        crop_size: Size of training crops
        num_crops: Number of crops to extract for training
        n_transforms: Number of augmentation transforms
        max_epochs: Maximum training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        use_gpu: Whether to use GPU if available
        visualize: Whether to save visualization of training samples
    """
    # Load and preprocess images
    data = load_and_preprocess_images(input_path)

    if visualize:
        visualize_sample(data, num_samples=3)

    # Create training crops
    print(f"\nCreating {num_crops} training crops of size {crop_size}x{crop_size}...")
    crops = create_training_crops(data, crop_size=crop_size, num_crops=num_crops)

    # Select a representative crop for training (or average them)
    # For simplicity, use the first crop
    training_crop = crops[0]

    print(f"Training crop shape: {training_crop.shape}")

    # Create DeepTrack dataset
    training_data = dt.Value(training_crop)

    # Initialize LodeSTAR model
    print("\nInitializing LodeSTAR model...")
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    lodestar = dl.LodeSTAR(
        n_transforms=n_transforms,
        optimizer=dl.Adam(lr=learning_rate),
    ).build()

    # Create dataset and dataloader
    dataset = dt.pytorch.Dataset(
        training_data,
        length=128,  # Number of training samples to generate
    )

    dataloader = dl.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
    )

    # Train the model
    print(f"\nTraining LodeSTAR for {max_epochs} epochs...")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print("-" * 60)

    lodestar_trainer = dl.Trainer(accelerator="auto", max_epochs=max_epochs)
    lodestar_trainer.fit(lodestar, dataloader)

    # Save the trained model
    print(f"\nSaving trained model to {output_path}...")
    torch.save(lodestar, output_path)

    print(f"\n✓ Training complete! Model saved to {output_path}")
    print(f"\nTo use this model for particle detection, run:")
    print(
        f'  uv run autolabeler.py --method lodestar --input "your_images/" --model-path {output_path}'
    )

    return lodestar


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a LodeSTAR model for particle tracking"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Directory containing training images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="lodestar_trained.pth",
        help="Output path for trained model (default: lodestar_trained.pth)",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=128,
        help="Size of training crops (default: 128)",
    )
    parser.add_argument(
        "--num-crops",
        type=int,
        default=10,
        help="Number of crops to extract (default: 10)",
    )
    parser.add_argument(
        "--n-transforms",
        type=int,
        default=4,
        help="Number of augmentation transforms (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size (default: 4)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip visualization of training samples",
    )

    args = parser.parse_args()

    train_lodestar(
        input_path=args.input,
        output_path=args.output,
        crop_size=args.crop_size,
        num_crops=args.num_crops,
        n_transforms=args.n_transforms,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_gpu=not args.no_gpu,
        visualize=not args.no_visualize,
    )
