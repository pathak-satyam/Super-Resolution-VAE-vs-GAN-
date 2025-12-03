"""
SRVAE Model Architecture
Super-Resolution Variational Autoencoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Encoder network for VAE"""

    def __init__(self, in_channels=3, latent_dim=128, img_size=64):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 64, 4, 2, 1)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        # Calculate flattened size
        self.flat_size = 512 * (img_size // 16) * (img_size // 16)

        # Latent space
        self.fc_mu = nn.Linear(self.flat_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_size, latent_dim)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class Decoder(nn.Module):
    """Decoder network for VAE"""

    def __init__(self, out_channels=3, latent_dim=128, img_size=64):
        super(Decoder, self).__init__()
        self.img_size = img_size
        self.init_size = img_size // 16

        # Linear layer to expand latent vector
        self.fc = nn.Linear(latent_dim, 512 * self.init_size * self.init_size)

        # Transposed convolutions
        self.deconv1 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(64, out_channels, 4, 2, 1)

        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 512, self.init_size, self.init_size)

        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = torch.tanh(self.deconv4(x))

        return x


class SRVAE(nn.Module):
    """Super-Resolution VAE with dual encoder-decoder architecture"""

    def __init__(self, latent_dim=128, lr_size=64, hr_size=256):
        super(SRVAE, self).__init__()

        self.latent_dim = latent_dim
        self.lr_size = lr_size
        self.hr_size = hr_size

        # HR network (High Resolution)
        self.hr_encoder = Encoder(in_channels=3, latent_dim=latent_dim, img_size=hr_size)
        self.hr_decoder = Decoder(out_channels=3, latent_dim=latent_dim, img_size=hr_size)

        # LR network (Low Resolution)
        self.lr_encoder = Encoder(in_channels=3, latent_dim=latent_dim, img_size=lr_size)
        self.lr_decoder = Decoder(out_channels=3, latent_dim=latent_dim, img_size=lr_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights properly to prevent NaN"""
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, hr_img, lr_img):
        """Forward pass with both HR and LR images"""
        # HR path
        hr_mu, hr_logvar = self.hr_encoder(hr_img)
        hr_z = self.reparameterize(hr_mu, hr_logvar)
        hr_recon = self.hr_decoder(hr_z)

        # LR path
        lr_mu, lr_logvar = self.lr_encoder(lr_img)
        lr_z = self.reparameterize(lr_mu, lr_logvar)
        lr_recon = self.lr_decoder(lr_z)

        # Super-resolution: LR recon -> HR encoder -> HR decoder
        lr_recon_upsampled = F.interpolate(lr_recon, size=self.hr_size, mode='bilinear', align_corners=False)
        sr_mu, sr_logvar = self.hr_encoder(lr_recon_upsampled)
        sr_z = self.reparameterize(sr_mu, sr_logvar)
        sr_img = self.hr_decoder(sr_z)

        return {
            'hr_recon': hr_recon,
            'hr_mu': hr_mu,
            'hr_logvar': hr_logvar,
            'lr_recon': lr_recon,
            'lr_mu': lr_mu,
            'lr_logvar': lr_logvar,
            'sr_img': sr_img,
            'sr_mu': sr_mu,
            'sr_logvar': sr_logvar
        }

    def super_resolve(self, lr_img):
        """Super-resolve a low-resolution image (inference)"""
        self.eval()
        with torch.no_grad():
            lr_mu, lr_logvar = self.lr_encoder(lr_img)
            lr_z = self.reparameterize(lr_mu, lr_logvar)
            lr_recon = self.lr_decoder(lr_z)

            lr_recon_upsampled = F.interpolate(lr_recon, size=self.hr_size, mode='bilinear', align_corners=False)
            sr_mu, _ = self.hr_encoder(lr_recon_upsampled)
            sr_img = self.hr_decoder(sr_mu)

        return sr_img