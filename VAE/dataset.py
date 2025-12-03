"""
Dataset loader for SRVAE training
Handles HR/LR image pair generation
"""

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    """Dataset for loading HR images and creating corresponding LR versions"""

    def __init__(self, hr_dir, hr_size=256, lr_size=64, max_images=None):
        """
        Args:
            hr_dir: Directory containing high-resolution images
            hr_size: Size of high-resolution images
            lr_size: Size of low-resolution images
            max_images: Optional limit on dataset size
        """
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
            import random
            random.seed(42)  # Reproducible subset
            self.image_files = random.sample(self.image_files, max_images)
            print(f"Using subset of {max_images} images from {len(os.listdir(hr_dir))} total")

        # Transforms for HR images
        self.hr_transform = transforms.Compose([
            transforms.Resize((hr_size, hr_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # Transforms for LR images
        self.lr_transform = transforms.Compose([
            transforms.Resize((lr_size, lr_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """Load and transform image pair"""
        img_path = os.path.join(self.hr_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        hr_img = self.hr_transform(image)
        lr_img = self.lr_transform(image)

        return hr_img, lr_img