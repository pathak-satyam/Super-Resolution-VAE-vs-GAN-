"""
Loss functions for SRVAE training
"""

import torch
import torch.nn.functional as F


def vae_loss(recon_x, x, mu, logvar, kl_weight=0.00001):
    """
    VAE loss = Reconstruction loss + KL divergence

    Args:
        recon_x: Reconstructed images
        x: Original images
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        kl_weight: Weight for KL divergence term

    Returns:
        total_loss: Combined loss
        recon_loss: Reconstruction loss component
        kl_div: KL divergence component
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='mean') * x.size(0)

    # Clamp logvar to prevent numerical instability
    logvar = torch.clamp(logvar, min=-10, max=10)

    # KL divergence
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    total_loss = recon_loss + kl_weight * kl_div

    return total_loss, recon_loss, kl_div


def adversarial_loss(encoder, real_imgs, fake_imgs):
    """
    Adversarial loss using encoder as discriminator

    Args:
        encoder: Encoder network used as discriminator
        real_imgs: Real images
        fake_imgs: Generated/fake images

    Returns:
        d_loss: Discriminator loss
    """
    # Encode real images
    real_mu, real_logvar = encoder(real_imgs)

    # Encode fake images (detached to avoid backprop through generator)
    fake_mu, fake_logvar = encoder(fake_imgs.detach())

    # Discriminator tries to separate real from fake based on latent space
    # Want real latents to be close to prior (N(0,1))
    real_score = -torch.mean(real_mu.pow(2))

    # Want fake latents to be far from prior
    fake_score = torch.mean(fake_mu.pow(2))

    d_loss = real_score + fake_score

    return d_loss