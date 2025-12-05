"""
SRGAN Evaluation
Evaluate trained SRGAN model using PSNR, SSIM, and MSE metrics
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from srgan_model import Generator
from dataset import ImageDataset


def calculate_psnr(img1, img2, max_value=1.0):
    """Calculate Peak Signal-to-Noise Ratio (PSNR)"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_value / torch.sqrt(mse))


def calculate_ssim(img1, img2, window_size=11, size_average=True):
    """Calculate Structural Similarity Index (SSIM)"""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = torch.nn.functional.avg_pool2d(img1, window_size, 1, padding=window_size // 2)
    mu2 = torch.nn.functional.avg_pool2d(img2, window_size, 1, padding=window_size // 2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.nn.functional.avg_pool2d(img1 * img1, window_size, 1,
                                               padding=window_size // 2) - mu1_sq
    sigma2_sq = torch.nn.functional.avg_pool2d(img2 * img2, window_size, 1,
                                               padding=window_size // 2) - mu2_sq
    sigma12 = torch.nn.functional.avg_pool2d(img1 * img2, window_size, 1,
                                             padding=window_size // 2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def evaluate_model(generator, dataloader, device):
    """Evaluate SRGAN generator on dataset"""
    generator.eval()

    total_psnr = 0
    total_ssim = 0
    total_mse = 0
    num_batches = 0

    with torch.no_grad():
        for hr_imgs, lr_imgs in tqdm(dataloader, desc="Evaluating"):
            hr_imgs = hr_imgs.to(device)
            lr_imgs = lr_imgs.to(device)

            # Generate SR images
            sr_imgs = generator(lr_imgs)

            # Denormalize from [-1, 1] to [0, 1]
            hr_imgs = torch.clamp((hr_imgs + 1) / 2, 0, 1)
            sr_imgs = torch.clamp((sr_imgs + 1) / 2, 0, 1)

            # Calculate metrics
            psnr = calculate_psnr(sr_imgs, hr_imgs)
            ssim = calculate_ssim(sr_imgs, hr_imgs)
            mse = torch.mean((sr_imgs - hr_imgs) ** 2)

            total_psnr += psnr.item()
            total_ssim += ssim.item()
            total_mse += mse.item()
            num_batches += 1

    return {
        'psnr': total_psnr / num_batches,
        'ssim': total_ssim / num_batches,
        'mse': total_mse / num_batches
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate SRGAN Model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained generator model')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to test data directory')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images to evaluate')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for evaluation')

    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load generator
    print(f"Loading model from {args.model}...")
    generator = Generator(num_residual_blocks=16).to(device)

    checkpoint = torch.load(args.model, map_location=device)
    if isinstance(checkpoint, dict) and 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        generator.load_state_dict(checkpoint)

    print("Model loaded successfully\n")
    # Create dataset
    print(f"Loading test data from {args.data}...")
    dataset = ImageDataset(
        hr_dir=args.data,
        hr_size=256,
        lr_size=64,
        max_images=args.max_images
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Test dataset size: {len(dataset)} images\n")

    # Evaluate
    print("Evaluating model...")
    metrics = evaluate_model(generator, dataloader, device)

    # Print results
    print("\n" + "=" * 70)
    print("SRGAN EVALUATION RESULTS")
    print("=" * 70)
    print(f"PSNR: {metrics['psnr']:.2f} dB")
    print(f"SSIM: {metrics['ssim']:.4f}")
    print(f"MSE:  {metrics['mse']:.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()