"""
SRVAE Training Script
Main training loop for Super-Resolution VAE
"""

import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import SRVAE
from dataset import ImageDataset
from loss import vae_loss, adversarial_loss


def train_srvae(
        hr_data_dir,
        num_epochs=100,
        batch_size=8,
        lr=0.0001,
        latent_dim=128,
        lr_size=64,
        hr_size=256,
        kl_weight=0.00001,
        adv_weight=0.0001,
        save_dir='./srvae_outputs',
        max_images=None
):

    # Create save directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'samples'), exist_ok=True)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Create dataset and dataloader
    dataset = ImageDataset(hr_data_dir, hr_size=hr_size, lr_size=lr_size, max_images=max_images)

    num_workers = min(4, batch_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2
    )

    print(f"\nDataset Configuration:")
    print(f"  Images: {len(dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Workers: {num_workers}")
    print(f"  Batches per epoch: {len(dataloader)}")

    # Initialize model
    model = SRVAE(latent_dim=latent_dim, lr_size=lr_size, hr_size=hr_size).to(device)
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # Training history
    history = {
        'hr_loss': [],
        'lr_loss': [],
        'sr_loss': [],
        'total_loss': []
    }

    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70 + "\n")

    for epoch in range(num_epochs):
        model.train()
        epoch_hr_loss = 0
        epoch_lr_loss = 0
        epoch_sr_loss = 0
        epoch_total_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch_idx, (hr_imgs, lr_imgs) in enumerate(pbar):
            hr_imgs = hr_imgs.to(device)
            lr_imgs = lr_imgs.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(hr_imgs, lr_imgs)

            # Calculate losses
            hr_loss, hr_recon, hr_kl = vae_loss(
                outputs['hr_recon'], hr_imgs,
                outputs['hr_mu'], outputs['hr_logvar'],
                kl_weight=kl_weight
            )

            lr_loss, lr_recon, lr_kl = vae_loss(
                outputs['lr_recon'], lr_imgs,
                outputs['lr_mu'], outputs['lr_logvar'],
                kl_weight=kl_weight
            )

            # Super-resolution loss
            sr_recon_loss = F.mse_loss(outputs['sr_img'], hr_imgs, reduction='mean') * hr_imgs.size(0)
            sr_logvar_clamped = torch.clamp(outputs['sr_logvar'], min=-10, max=10)
            sr_kl = -0.5 * torch.sum(
                1 + sr_logvar_clamped - outputs['sr_mu'].pow(2) - sr_logvar_clamped.exp()
            )
            sr_loss = sr_recon_loss + kl_weight * sr_kl

            # Adversarial loss
            adv_loss = adversarial_loss(model.hr_encoder, hr_imgs, outputs['sr_img'])

            # Total loss
            total_loss = hr_loss + lr_loss + sr_loss + adv_weight * adv_loss

            # Backward pass
            total_loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Check for NaN
            if torch.isnan(total_loss):
                print("\nWARNING: NaN detected! Skipping this batch...")
                continue

            # Update metrics
            epoch_hr_loss += hr_loss.item()
            epoch_lr_loss += lr_loss.item()
            epoch_sr_loss += sr_loss.item()
            epoch_total_loss += total_loss.item()

            # Update progress bar
            pbar.set_postfix({
                'HR': f'{hr_loss.item():.2f}',
                'LR': f'{lr_loss.item():.2f}',
                'SR': f'{sr_loss.item():.2f}'
            })

        # Calculate average losses
        avg_hr_loss = epoch_hr_loss / len(dataloader)
        avg_lr_loss = epoch_lr_loss / len(dataloader)
        avg_sr_loss = epoch_sr_loss / len(dataloader)
        avg_total_loss = epoch_total_loss / len(dataloader)

        # Store history
        history['hr_loss'].append(avg_hr_loss)
        history['lr_loss'].append(avg_lr_loss)
        history['sr_loss'].append(avg_sr_loss)
        history['total_loss'].append(avg_total_loss)

        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"HR: {avg_hr_loss:.2f}, LR: {avg_lr_loss:.2f}, SR: {avg_sr_loss:.2f}, "
              f"Total: {avg_total_loss:.2f}")

        # Update learning rate
        scheduler.step()

        # Save sample images every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_samples(model, dataloader, epoch, hr_size, save_dir, device)

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, history, save_dir)

    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'srvae_final.pth'))

    # Plot and save training history
    plot_history(history, save_dir)

    print(f"\n{'=' * 70}")
    print("Training Complete!")
    print(f"{'=' * 70}")
    print(f"Model saved to: {save_dir}/srvae_final.pth")
    print(f"Training history: {save_dir}/training_history.png")
    print(f"Sample images: {save_dir}/samples/")

    return model, history


def save_samples(model, dataloader, epoch, hr_size, save_dir, device):
    """Save sample super-resolution results"""
    model.eval()
    with torch.no_grad():
        hr_sample, lr_sample = next(iter(dataloader))
        hr_sample = hr_sample[:4].to(device)
        lr_sample = lr_sample[:4].to(device)

        outputs = model(hr_sample, lr_sample)
        sr_sample = outputs['sr_img']

        # Denormalize images
        hr_sample = (hr_sample + 1) / 2
        lr_sample = (lr_sample + 1) / 2
        sr_sample = (sr_sample + 1) / 2

        # Create comparison grid: [LR upsampled | SR | HR ground truth]
        comparison = torch.cat([
            F.interpolate(lr_sample, size=hr_size, mode='bilinear', align_corners=False),
            sr_sample,
            hr_sample
        ], dim=0)

        grid = make_grid(comparison, nrow=4, padding=2)
        save_image(grid, os.path.join(save_dir, 'samples', f'epoch_{epoch + 1}.png'))


def save_checkpoint(model, optimizer, epoch, history, save_dir):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }
    checkpoint_path = os.path.join(save_dir, 'checkpoints', f'checkpoint_epoch_{epoch + 1}.pth')
    torch.save(checkpoint, checkpoint_path)


def plot_history(history, save_dir):
    """Plot and save training history"""
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(history['hr_loss'])
    plt.title('HR Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(history['lr_loss'])
    plt.title('LR Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(history['sr_loss'])
    plt.title('SR Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(history['total_loss'])
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150)
    plt.close()


if __name__ == "__main__":
    import glob

    # Configuration
    DATA_PATH = r"C:\Users\satya\OneDrive\Documents\SuperResolution\VAE\data\img_align_celeba"

    print("=" * 70)
    print("SRVAE Training - Face Super-Resolution")
    print("=" * 70)

    # Check if dataset exists
    if not os.path.exists(DATA_PATH):
        print(f"\nERROR: Dataset not found at: {DATA_PATH}")
        print("Please update DATA_PATH in this script!")
        exit(1)

    # Count images
    all_images = glob.glob(os.path.join(DATA_PATH, "*.jpg")) + \
                 glob.glob(os.path.join(DATA_PATH, "*.png"))
    num_images = len(all_images)

    print(f"\nFound {num_images:,} images in dataset")

    # Dataset size options
    print("\n" + "=" * 70)
    print("Dataset Size Options:")
    print("=" * 70)
    print("1. Use ALL images (may take 30-40+ hours)")
    print("2. Use 30,000 images (~6-8 hours) - RECOMMENDED")
    print("3. Use 10,000 images (~2-3 hours) - Quick training")
    print("4. Use 5,000 images (~1-2 hours) - Fast testing")
    print("5. Custom amount")

    subset_choice = input("\nChoose option (1-5): ").strip()

    if subset_choice == "1":
        max_images = num_images
        batch_size = 4
        num_epochs = 50
    elif subset_choice == "2":
        max_images = 30000
        batch_size = 8
        num_epochs = 80
    elif subset_choice == "3":
        max_images = 10000
        batch_size = 8
        num_epochs = 100
    elif subset_choice == "4":
        max_images = 5000
        batch_size = 8
        num_epochs = 100
    elif subset_choice == "5":
        max_images_input = input("Enter number of images: ").replace(",", "").strip()
        max_images = int(max_images_input)
        batch_size = 8 if max_images <= 20000 else 4
        num_epochs_input = input("Enter number of epochs: ").strip()
        num_epochs = int(num_epochs_input)
    else:
        print("Invalid choice. Using default: 10,000 images")
        max_images = 10000
        batch_size = 8
        num_epochs = 100

    # Display configuration
    print(f"\nTraining Configuration:")
    print(f"  Images: {max_images:,}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Resolution: 64 -> 256")
    print(f"  Learning Rate: 0.0001")
    print(f"  Latent Dim: 128")

    response = input("\nStart training? (y/n): ").strip().lower()
    if response != 'y':
        print("Training cancelled.")
        exit(0)

    print("\nInitializing training...\n")

    try:
        model, history = train_srvae(
            hr_data_dir=DATA_PATH,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=0.0001,
            latent_dim=128,
            lr_size=64,
            hr_size=256,
            kl_weight=0.00001,
            adv_weight=0.0001,
            save_dir='./srvae_outputs',
            max_images=max_images
        )

        print("\nTraining completed successfully!")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Latest checkpoint saved in: ./srvae_outputs/checkpoints/")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()