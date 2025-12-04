"""
SRGAN Model Architecture
Implements Generator, Discriminator, and VGG Perceptual Loss for Super-Resolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResidualBlock(nn.Module):
    """Residual block with skip connection"""

    def __init__(self, channels=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.prelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out + residual


class UpsampleBlock(nn.Module):
    """Upsampling block using pixel shuffle"""

    def __init__(self, in_channels, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (scale_factor ** 2),
                              kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class Generator(nn.Module):
    """
    SRGAN Generator Network
    Upscales 64x64x3 LR images to 256x256x3 SR images (4x upscaling)

    Architecture:
        - Initial feature extraction (9x9 conv)
        - Residual blocks for deep feature learning
        - Upsampling blocks (2x -> 2x = 4x total)
        - Final reconstruction layer
    """

    def __init__(self, num_residual_blocks=16):
        super(Generator, self).__init__()

        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )

        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residual_blocks)]
        )

        # Post-residual convolution
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )

        # Upsampling layers (2x -> 2x = 4x total)
        self.upsample1 = UpsampleBlock(64, scale_factor=2)  # 64x64 -> 128x128
        self.upsample2 = UpsampleBlock(64, scale_factor=2)  # 128x128 -> 256x256

        # Final output convolution
        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, padding=4)

    def forward(self, x):
        # Initial feature extraction
        out1 = self.conv1(x)

        # Residual learning
        out = self.residual_blocks(out1)
        out2 = self.conv2(out)

        # Skip connection from input features
        out = out1 + out2

        # Upsampling
        out = self.upsample1(out)
        out = self.upsample2(out)

        # Final output (tanh for [-1, 1] range)
        out = torch.tanh(self.conv3(out))

        return out


class DiscriminatorBlock(nn.Module):
    """Discriminator convolutional block"""

    def __init__(self, in_channels, out_channels, stride=1, use_bn=True):
        super(DiscriminatorBlock, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3,
                            stride=stride, padding=1)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Discriminator(nn.Module):
    """
    SRGAN Discriminator Network
    Classifies 256x256 images as real (HR) or fake (SR)

    Architecture:
        - Convolutional feature extraction
        - Strided convolutions for downsampling
        - Global average pooling
        - Dense classification layers
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        self.features = nn.Sequential(
            # Input: 256x256x3
            DiscriminatorBlock(3, 64, stride=1, use_bn=False),
            DiscriminatorBlock(64, 64, stride=2),  # 128x128

            DiscriminatorBlock(64, 128, stride=1),
            DiscriminatorBlock(128, 128, stride=2),  # 64x64

            DiscriminatorBlock(128, 256, stride=1),
            DiscriminatorBlock(256, 256, stride=2),  # 32x32

            DiscriminatorBlock(256, 512, stride=1),
            DiscriminatorBlock(512, 512, stride=2),  # 16x16
        )

        # Dense layers
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual Loss using VGG19 features
    Compares high-level feature representations instead of raw pixels

    Uses pre-trained VGG19 up to conv5_4 layer
    """

    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()

        # Load pre-trained VGG19
        vgg = models.vgg19(pretrained=True).features

        # Use features up to conv5_4 (before maxpool)
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:36]).eval()

        # Freeze VGG parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Normalization for ImageNet pre-trained model
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, sr, hr):
        # Denormalize from [-1, 1] to [0, 1]
        sr = (sr + 1) / 2
        hr = (hr + 1) / 2

        # Normalize for VGG (ImageNet stats)
        sr = (sr - self.mean) / self.std
        hr = (hr - self.mean) / self.std

        # Extract features
        sr_features = self.feature_extractor(sr)
        hr_features = self.feature_extractor(hr)

        # MSE loss on features
        loss = F.mse_loss(sr_features, hr_features)
        return loss


class SRGAN(nn.Module):
    """Complete SRGAN model with Generator and Discriminator"""

    def __init__(self, num_residual_blocks=16):
        super(SRGAN, self).__init__()
        self.generator = Generator(num_residual_blocks)
        self.discriminator = Discriminator()
        self.perceptual_loss = VGGPerceptualLoss()

    def forward(self, lr_img):
        """Generate SR image from LR input"""
        return self.generator(lr_img)


def test_srgan():
    """Test the SRGAN architecture"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create models
    generator = Generator(num_residual_blocks=16).to(device)
    discriminator = Discriminator().to(device)
    perceptual_loss = VGGPerceptualLoss().to(device)

    # Test input
    lr_img = torch.randn(2, 3, 64, 64).to(device)
    hr_img = torch.randn(2, 3, 256, 256).to(device)

    # Test generator
    print("Testing Generator...")
    sr_img = generator(lr_img)
    print(f"  Input shape: {lr_img.shape}")
    print(f"  Output shape: {sr_img.shape}")
    assert sr_img.shape == (2, 3, 256, 256), "Generator output shape mismatch"
    print("  Generator works correctly")

    # Test discriminator
    print("\nTesting Discriminator...")
    real_output = discriminator(hr_img)
    fake_output = discriminator(sr_img.detach())
    print(f"  Real output: {real_output.shape} (values: {real_output.mean().item():.3f})")
    print(f"  Fake output: {fake_output.shape} (values: {fake_output.mean().item():.3f})")
    assert real_output.shape == (2, 1), "Discriminator output shape mismatch"
    print("  Discriminator works correctly")

    # Test perceptual loss
    print("\nTesting Perceptual Loss...")
    perc_loss = perceptual_loss(sr_img, hr_img)
    print(f"  Perceptual loss: {perc_loss.item():.4f}")
    print("  Perceptual loss works correctly")

    # Count parameters
    gen_params = sum(p.numel() for p in generator.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())

    print("\n" + "=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)
    print(f"Generator parameters: {gen_params:,}")
    print(f"Discriminator parameters: {disc_params:,}")
    print(f"Total parameters: {gen_params + disc_params:,}")
    print("=" * 70)

    return generator, discriminator


if __name__ == "__main__":
    print("=" * 70)
    print("SRGAN Architecture Test")
    print("=" * 70)
    print()

    test_srgan()

    print("\nAll tests passed. SRGAN architecture is ready for training.")