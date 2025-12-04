"""
SRGAN Training Script
Train Super-Resolution GAN on CelebA dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import os
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob

from srgan_model import Generator, Discriminator, VGGPerceptualLoss


class ImageDataset(Dataset):
    """Dataset for loading HR images and creating LR versions"""

    def __init__(self, hr_dir, hr_size=256, lr_size=64, max_images=None):
        self.hr_dir = hr_dir
        self.hr_size = hr_size
        self.lr_size = lr_size

        self.image_files = [f for f in os.listdir(hr_dir)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        if max_images is not None and max_images < len(self.image_files):
            import random
            random.seed(42)
            self.image_files = random.sample(self.image_files, max_images)

        self.hr_transform = transforms.Compose([
            transforms.Resize((hr_size, hr_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

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
        return self.hr_transform(image), self.lr_transform(image)


def train_srgan(
        hr_data_dir,
        num_epochs=100,
        batch_size=8,
        lr_g=0.0001,
        lr_d=0.0001,
        num_residual_blocks=16,
        lambda_pixel=1.0,
        lambda_perceptual=0.006,
        lambda_adversarial=0.001,
        save_dir='./srgan_outputs',
        max_images=None
):
    """
    Train SRGAN model

    Args:
        hr_data_dir: Path to high-resolution images
        num_epochs: Number of training epochs
        batch_size: Batch size
        lr_g: Generator learning rate
        lr_d: Discriminator learning rate
        num_residual_blocks: Number of residual blocks in generator
        lambda_pixel: Weight for pixel-wise MSE loss
        lambda_perceptual: Weight for perceptual (VGG) loss
        lambda_adversarial: Weight for adversarial loss
        save_dir: Directory to save outputs
        max_images: Maximum number of images to use (None = all)
    """
    # Create save directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'samples'), exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Create dataset and dataloader
    dataset = ImageDataset(hr_data_dir, hr_size=256, lr_size=64, max_images=max_images)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
        persistent_workers=True, prefetch_factor=2
    )

    print(f"\nDataset size: {len(dataset)} images")
    print(f"Batch size: {batch_size}")
    print(f"Batches per epoch: {len(dataloader)}")

    # Initialize models
    generator = Generator(num_residual_blocks=num_residual_blocks).to(device)
    discriminator = Discriminator().to(device)
    perceptual_loss_fn = VGGPerceptualLoss().to(device)

    # Count parameters
    gen_params = sum(p.numel() for p in generator.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    print(f"\nGenerator parameters: {gen_params:,}")
    print(f"Discriminator parameters: {disc_params:,}")
    print(f"Total parameters: {gen_params + disc_params:,}")

    # Optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.9, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.9, 0.999))

    # Learning rate schedulers
    scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=30, gamma=0.5)
    scheduler_d = optim.lr_scheduler.StepLR(optimizer_d, step_size=30, gamma=0.5)

    # Loss functions
    criterion_pixel = nn.MSELoss()
    criterion_adversarial = nn.BCELoss()

    # Training history
    history = {
        'g_loss': [],
        'd_loss': [],
        'pixel_loss': [],
        'perceptual_loss': [],
        'adversarial_loss': []
    }

    # Training loop
    print("\n" + "=" * 70)
    print("STARTING SRGAN TRAINING")
    print("=" * 70)
    print(f"Epochs: {num_epochs}")
    print(f"Loss weights: Pixel={lambda_pixel}, Perceptual={lambda_perceptual}, Adversarial={lambda_adversarial}")
    print("=" * 70 + "\n")

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()

        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_pixel_loss = 0
        epoch_perc_loss = 0
        epoch_adv_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch_idx, (hr_imgs, lr_imgs) in enumerate(pbar):
            hr_imgs = hr_imgs.to(device)
            lr_imgs = lr_imgs.to(device)
            batch_size_current = hr_imgs.size(0)

            # Labels for adversarial loss
            real_labels = torch.ones(batch_size_current, 1).to(device)
            fake_labels = torch.zeros(batch_size_current, 1).to(device)

            # Train Discriminator
            optimizer_d.zero_grad()

            # Generate fake images
            sr_imgs = generator(lr_imgs)

            # Real images
            real_output = discriminator(hr_imgs)
            d_loss_real = criterion_adversarial(real_output, real_labels)

            # Fake images
            fake_output = discriminator(sr_imgs.detach())
            d_loss_fake = criterion_adversarial(fake_output, fake_labels)

            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()

            # Generate SR images
            sr_imgs = generator(lr_imgs)

            # Pixel-wise loss (MSE)
            pixel_loss = criterion_pixel(sr_imgs, hr_imgs)

            # Perceptual loss (VGG features)
            perceptual_loss = perceptual_loss_fn(sr_imgs, hr_imgs)

            # Adversarial loss (fool discriminator)
            fake_output = discriminator(sr_imgs)
            adversarial_loss = criterion_adversarial(fake_output, real_labels)

            # Total generator loss
            g_loss = (lambda_pixel * pixel_loss +
                      lambda_perceptual * perceptual_loss +
                      lambda_adversarial * adversarial_loss)

            g_loss.backward()
            optimizer_g.step()

            # Accumulate losses
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_pixel_loss += pixel_loss.item()
            epoch_perc_loss += perceptual_loss.item()
            epoch_adv_loss += adversarial_loss.item()

            # Update progress bar
            pbar.set_postfix({
                'G': f'{g_loss.item():.3f}',
                'D': f'{d_loss.item():.3f}',
                'Pix': f'{pixel_loss.item():.3f}'
            })

        # Calculate average losses
        avg_g_loss = epoch_g_loss / len(dataloader)
        avg_d_loss = epoch_d_loss / len(dataloader)
        avg_pixel_loss = epoch_pixel_loss / len(dataloader)
        avg_perc_loss = epoch_perc_loss / len(dataloader)
        avg_adv_loss = epoch_adv_loss / len(dataloader)

        history['g_loss'].append(avg_g_loss)
        history['d_loss'].append(avg_d_loss)
        history['pixel_loss'].append(avg_pixel_loss)
        history['perceptual_loss'].append(avg_perc_loss)
        history['adversarial_loss'].append(avg_adv_loss)

        print(f"Epoch {epoch + 1}/{num_epochs} - G: {avg_g_loss:.3f}, D: {avg_d_loss:.3f}, "
              f"Pixel: {avg_pixel_loss:.3f}, Perc: {avg_perc_loss:.3f}, Adv: {avg_adv_loss:.3f}")

        # Update learning rates
        scheduler_g.step()
        scheduler_d.step()

        # Save sample images
        if (epoch + 1) % 5 == 0:
            generator.eval()
            with torch.no_grad():
                hr_sample, lr_sample = next(iter(dataloader))
                hr_sample = hr_sample[:4].to(device)
                lr_sample = lr_sample[:4].to(device)

                sr_sample = generator(lr_sample)

                # Denormalize
                hr_sample = torch.clamp((hr_sample + 1) / 2, 0, 1)
                lr_sample = torch.clamp((lr_sample + 1) / 2, 0, 1)
                sr_sample = torch.clamp((sr_sample + 1) / 2, 0, 1)

                # Upscale LR for comparison
                lr_upscaled = F.interpolate(lr_sample, size=256, mode='bilinear', align_corners=False)

                # Create comparison grid
                comparison = torch.cat([lr_upscaled, sr_sample, hr_sample], dim=0)
                grid = make_grid(comparison, nrow=4, padding=2)
                save_image(grid, os.path.join(save_dir, 'samples', f'epoch_{epoch + 1}.png'))

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'history': history
            }
            torch.save(checkpoint, os.path.join(save_dir, 'checkpoints', f'checkpoint_epoch_{epoch + 1}.pth'))

    # Save final models
    torch.save(generator.state_dict(), os.path.join(save_dir, 'generator_final.pth'))
    torch.save(discriminator.state_dict(), os.path.join(save_dir, 'discriminator_final.pth'))

    # Plot training history
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.plot(history['g_loss'])
    plt.title('Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(2, 3, 2)
    plt.plot(history['d_loss'])
    plt.title('Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(2, 3, 3)
    plt.plot(history['pixel_loss'])
    plt.title('Pixel Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(2, 3, 4)
    plt.plot(history['perceptual_loss'])
    plt.title('Perceptual Loss (VGG)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(2, 3, 5)
    plt.plot(history['adversarial_loss'])
    plt.title('Adversarial Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(2, 3, 6)
    plt.plot(history['g_loss'], label='Generator')
    plt.plot(history['d_loss'], label='Discriminator')
    plt.title('G vs D Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()

    print(f"\nTraining complete. Models saved to {save_dir}")
    return generator, discriminator, history


if __name__ == "__main__":
    import sys

    # Configuration
    DATA_PATH = r"C:\Users\satya\OneDrive\Documents\SuperResolution\VAE\data\img_align_celeba"

    if len(sys.argv) > 1:
        DATA_PATH = sys.argv[1]

    print("=" * 70)
    print("SRGAN Training - CelebA Face Super-Resolution")
    print("=" * 70)

    if not os.path.exists(DATA_PATH):
        print(f"\nDataset not found: {DATA_PATH}")
        print("Please update DATA_PATH in this script or pass as argument")
        sys.exit(1)

    # Count images
    all_images = glob.glob(os.path.join(DATA_PATH, "*.jpg")) + \
                 glob.glob(os.path.join(DATA_PATH, "*.png"))
    num_images = len(all_images)

    print(f"\nFound {num_images:,} images")

    # Dataset size options
    print("\n" + "=" * 70)
    print("DATASET SIZE OPTIONS:")
    print("=" * 70)
    print("1. Use 30,000 images (~6-8 hours)")
    print("2. Use 10,000 images (~2-3 hours) - RECOMMENDED")
    print("3. Use 5,000 images (~1-2 hours)")
    print("4. Custom amount")

    choice = input("\nChoose option (1-4): ").strip()

    if choice == "1":
        max_images = 30000
        num_epochs = 80
    elif choice == "2":
        max_images = 10000
        num_epochs = 100
    elif choice == "3":
        max_images = 5000
        num_epochs = 100
    elif choice == "4":
        max_images = int(input("Enter number of images: ").replace(",", ""))
        num_epochs = int(input("Enter number of epochs: "))
    else:
        max_images = 10000
        num_epochs = 100

    print(f"\nConfiguration:")
    print(f"   Images: {max_images:,}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch Size: 8")
    print(f"   Resolution: 64 -> 256 (4x)")

    response = input("\nStart training? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        sys.exit(0)

    print("\nStarting SRGAN training...\n")

    try:
        generator, discriminator, history = train_srgan(
            hr_data_dir=DATA_PATH,
            num_epochs=num_epochs,
            batch_size=8,
            lr_g=0.0001,
            lr_d=0.0001,
            num_residual_blocks=16,
            lambda_pixel=1.0,
            lambda_perceptual=0.006,
            lambda_adversarial=0.001,
            save_dir='./srgan_outputs',
            max_images=max_images
        )

        print("\nSRGAN training completed successfully")
        print(f"Generator: ./srgan_outputs/generator_final.pth")
        print(f"Discriminator: ./srgan_outputs/discriminator_final.pth")
        print(f"History: ./srgan_outputs/training_history.png")
        print(f"Samples: ./srgan_outputs/samples/")

    except KeyboardInterrupt:
        print("\nTraining interrupted")
        print("Latest checkpoint saved")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()