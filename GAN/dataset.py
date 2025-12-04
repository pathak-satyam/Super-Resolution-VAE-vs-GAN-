"""
Dataset Utilities
Data loading and preprocessing for SRGAN training
"""

import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    """
    Dataset for loading HR images and creating LR versions

    Args:
        hr_dir: Directory containing high-resolution images
        hr_size: Target high-resolution size (default: 256)
        lr_size: Target low-resolution size (default: 64)
        max_images: Maximum number of images to load (None = all)
        seed: Random seed for reproducibility
    """

    def __init__(self, hr_dir, hr_size=256, lr_size=64, max_images=None, seed=42):
        self.hr_dir = hr_dir
        self.hr_size = hr_size
        self.lr_size = lr_size

        # Get all image files
        self.image_files = [
            f for f in os.listdir(hr_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]

        # Limit dataset size if specified
        if max_images is not None and max_images < len(self.image_files):
            random.seed(seed)
            self.image_files = random.sample(self.image_files, max_images)

        # HR transform
        self.hr_transform = transforms.Compose([
            transforms.Resize((hr_size, hr_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # LR transform
        self.lr_transform = transforms.Compose([
            transforms.Resize((lr_size, lr_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.hr_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        hr_image = self.hr_transform(image)
        lr_image = self.lr_transform(image)

        return hr_image, lr_image


def denormalize(tensor):
    """
    Denormalize tensor from [-1, 1] to [0, 1]

    Args:
        tensor: Input tensor normalized to [-1, 1]

    Returns:
        Tensor denormalized to [0, 1]
    """
    return torch.clamp((tensor + 1) / 2, 0, 1)


def get_dataloader(data_dir, batch_size=8, hr_size=256, lr_size=64,
                   max_images=None, num_workers=4, shuffle=True):
    """
    Create DataLoader for SRGAN training

    Args:
        data_dir: Path to image directory
        batch_size: Batch size for training
        hr_size: High-resolution image size
        lr_size: Low-resolution image size
        max_images: Maximum number of images to use
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle the data

    Returns:
        DataLoader instance
    """
    dataset = ImageDataset(
        hr_dir=data_dir,
        hr_size=hr_size,
        lr_size=lr_size,
        max_images=max_images
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    return dataloader