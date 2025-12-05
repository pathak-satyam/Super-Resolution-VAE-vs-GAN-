"""
Super-Resolution Inference
Takes a blurry image and makes it sharp using trained models
"""

import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys

MODEL_TYPE = 'srgan'  # or 'vae'

if MODEL_TYPE == 'srgan':
    sys.path.append('C:/Users/satya/OneDrive/Documents/SuperResolution/GAN')
    from srgan_model import Generator
    MODEL_PATH = 'C:/Users/satya/OneDrive/Documents/SuperResolution/GAN/srgan_outputs/checkpoints/checkpoint_epoch_90.pth'
else:
    sys.path.append('C:/Users/satya/OneDrive/Documents/SuperResolution/VAE')
    from model import SRVAE
    MODEL_PATH = 'C:/Users/satya/OneDrive/Documents/SuperResolution/VAE/srvae_outputs/checkpoints/checkpoint_epoch_99.pth'

# Load image
INPUT_IMAGE = 'C:/Users/satya/OneDrive/Documents/SuperResolution/VAE/data/img_align_celeba/000798.jpg'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model
print(f"Loading {MODEL_TYPE.upper()} model...")
if MODEL_TYPE == 'srgan':
    model = Generator(num_residual_blocks=16).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['generator_state_dict'])
else:
    model = SRVAE(latent_dim=128, lr_size=64, hr_size=256).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
print("Model loaded!")

# Prepare image
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Simulate low-res input
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

image = Image.open(INPUT_IMAGE).convert('RGB')
lr_image = transform(image).unsqueeze(0).to(device)

# Generate high-res image
print("Generating super-resolution image...")
with torch.no_grad():
    if MODEL_TYPE == 'srgan':
        sr_image = model(lr_image)
    else:
        sr_image = model.super_resolve(lr_image)

# Denormalize
sr_image = torch.clamp((sr_image + 1) / 2, 0, 1)
sr_image = sr_image.squeeze(0).cpu()

# Convert to PIL
sr_image_pil = transforms.ToPILImage()(sr_image)

# Display results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(image.resize((64, 64), Image.BILINEAR).resize((256, 256), Image.BILINEAR))
axes[1].set_title('Blurry (64×64 → 256×256)')
axes[1].axis('off')

axes[2].imshow(sr_image_pil)
axes[2].set_title(f'{MODEL_TYPE.upper()} Super-Resolution')
axes[2].axis('off')

plt.tight_layout()
plt.savefig(f'super_resolution_result_{MODEL_TYPE}.png', dpi=300, bbox_inches='tight')
plt.show()

# Save the sharp image
sr_image_pil.save(f'super_resolved_{MODEL_TYPE}.png')
print(f"\n✓ Sharp image saved as 'super_resolved_{MODEL_TYPE}.png'")