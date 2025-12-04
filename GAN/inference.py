"""
SRGAN Inference
Test trained SRGAN model on new images
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
import argparse

from srgan_model import Generator


def load_image(image_path, size=64):
    """
    Load and preprocess a single image

    Args:
        image_path: Path to input image
        size: Size to resize image to

    Returns:
        Preprocessed image tensor
    """
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)


def super_resolve(generator, lr_image, device):
    """
    Generate super-resolution image

    Args:
        generator: Trained generator model
        lr_image: Low-resolution input tensor
        device: Torch device

    Returns:
        Super-resolved image tensor
    """
    generator.eval()

    with torch.no_grad():
        lr_image = lr_image.to(device)
        sr_image = generator(lr_image)

        # Denormalize from [-1, 1] to [0, 1]
        sr_image = torch.clamp((sr_image + 1) / 2, 0, 1)

    return sr_image


def compare_images(lr_image, sr_image, hr_image=None, save_path='comparison.png'):
    """
    Create comparison of LR, SR, and optionally HR images

    Args:
        lr_image: Low-resolution image tensor
        sr_image: Super-resolved image tensor
        hr_image: High-resolution ground truth (optional)
        save_path: Path to save comparison
    """
    # Denormalize LR
    lr_image = torch.clamp((lr_image + 1) / 2, 0, 1)

    # Upscale LR for visual comparison
    lr_upscaled = F.interpolate(lr_image, size=256, mode='bilinear',
                                align_corners=False)

    if hr_image is not None:
        hr_image = torch.clamp((hr_image + 1) / 2, 0, 1)
        images = torch.cat([lr_upscaled, sr_image, hr_image], dim=0)
    else:
        images = torch.cat([lr_upscaled, sr_image], dim=0)

    save_image(images, save_path, nrow=len(images))


def main():
    parser = argparse.ArgumentParser(description='SRGAN Inference')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained generator model')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input low-resolution image')
    parser.add_argument('--output', type=str, default='output.png',
                        help='Path to save output image')
    parser.add_argument('--compare', action='store_true',
                        help='Generate comparison image')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for inference')

    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load generator
    print(f"Loading model from {args.model}...")
    generator = Generator(num_residual_blocks=16).to(device)
    generator.load_state_dict(torch.load(args.model, map_location=device))
    print("Model loaded successfully")

    # Load input image
    print(f"Loading image from {args.input}...")
    lr_image = load_image(args.input, size=64)

    # Generate super-resolution
    print("Generating super-resolution image...")
    sr_image = super_resolve(generator, lr_image, device)

    # Save output
    if args.compare:
        compare_images(lr_image, sr_image, save_path=args.output)
        print(f"Comparison saved to {args.output}")
    else:
        save_image(sr_image, args.output)
        print(f"Super-resolved image saved to {args.output}")


if __name__ == "__main__":
    main()