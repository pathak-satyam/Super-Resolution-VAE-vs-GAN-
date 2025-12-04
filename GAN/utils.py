"""
Utilities
Helper functions for training, evaluation, and visualization
"""

import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid


def count_parameters(model):
    """
    Count trainable parameters in a model

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(epoch, generator, discriminator, optimizer_g, optimizer_d,
                    history, filepath):
    """
    Save training checkpoint

    Args:
        epoch: Current epoch number
        generator: Generator model
        discriminator: Discriminator model
        optimizer_g: Generator optimizer
        optimizer_d: Discriminator optimizer
        history: Training history dictionary
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
        'history': history
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, generator, discriminator, optimizer_g=None,
                    optimizer_d=None):
    """
    Load training checkpoint

    Args:
        filepath: Path to checkpoint file
        generator: Generator model
        discriminator: Discriminator model
        optimizer_g: Generator optimizer (optional)
        optimizer_d: Discriminator optimizer (optional)

    Returns:
        Tuple of (epoch, history)
    """
    checkpoint = torch.load(filepath)

    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

    if optimizer_g is not None:
        optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    if optimizer_d is not None:
        optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])

    return checkpoint['epoch'], checkpoint['history']


def save_sample_images(generator, dataloader, device, save_path, num_samples=4):
    """
    Generate and save sample super-resolution images

    Args:
        generator: Generator model
        dataloader: Data loader
        device: Torch device
        save_path: Path to save the image grid
        num_samples: Number of samples to generate
    """
    generator.eval()

    with torch.no_grad():
        hr_sample, lr_sample = next(iter(dataloader))
        hr_sample = hr_sample[:num_samples].to(device)
        lr_sample = lr_sample[:num_samples].to(device)

        # Generate SR images
        sr_sample = generator(lr_sample)

        # Denormalize from [-1, 1] to [0, 1]
        hr_sample = torch.clamp((hr_sample + 1) / 2, 0, 1)
        lr_sample = torch.clamp((lr_sample + 1) / 2, 0, 1)
        sr_sample = torch.clamp((sr_sample + 1) / 2, 0, 1)

        # Upscale LR for visual comparison
        lr_upscaled = F.interpolate(lr_sample, size=256, mode='bilinear',
                                    align_corners=False)

        # Create comparison grid: [LR_upscaled, SR, HR]
        comparison = torch.cat([lr_upscaled, sr_sample, hr_sample], dim=0)
        grid = make_grid(comparison, nrow=num_samples, padding=2)
        save_image(grid, save_path)


def plot_training_history(history, save_path):
    """
    Plot and save training history

    Args:
        history: Dictionary containing loss history
        save_path: Path to save the plot
    """
    plt.figure(figsize=(15, 10))

    # Generator Loss
    plt.subplot(2, 3, 1)
    plt.plot(history['g_loss'])
    plt.title('Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    # Discriminator Loss
    plt.subplot(2, 3, 2)
    plt.plot(history['d_loss'])
    plt.title('Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    # Pixel Loss
    plt.subplot(2, 3, 3)
    plt.plot(history['pixel_loss'])
    plt.title('Pixel Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    # Perceptual Loss
    plt.subplot(2, 3, 4)
    plt.plot(history['perceptual_loss'])
    plt.title('Perceptual Loss (VGG)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    # Adversarial Loss
    plt.subplot(2, 3, 5)
    plt.plot(history['adversarial_loss'])
    plt.title('Adversarial Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    # Combined G vs D
    plt.subplot(2, 3, 6)
    plt.plot(history['g_loss'], label='Generator')
    plt.plot(history['d_loss'], label='Discriminator')
    plt.title('Generator vs Discriminator')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def print_device_info():
    """Print information about the computing device"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    return device


def print_model_summary(generator, discriminator):
    """
    Print summary of model architectures

    Args:
        generator: Generator model
        discriminator: Discriminator model
    """
    gen_params = count_parameters(generator)
    disc_params = count_parameters(discriminator)

    print("\n" + "=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)
    print(f"Generator parameters: {gen_params:,}")
    print(f"Discriminator parameters: {disc_params:,}")
    print(f"Total parameters: {gen_params + disc_params:,}")
    print("=" * 70)