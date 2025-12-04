# Super-Resolution Comparison: VAE vs GAN

Face image super-resolution using deep learning approaches. This project compares two different architectures for upscaling 64×64 pixel face images to 256×256 pixels (4x super-resolution) on the CelebA dataset.

## Project Structure

```
super-resolution-comparison/
├── VAE/              # Variational Autoencoder approach
│   ├── model.py
│   ├── dataset.py
│   ├── loss.py
│   ├── train.py
│   └── requirements.txt
└── GAN/              # Generative Adversarial Network approach
    ├── srgan_model.py
    ├── train_srgan.py
    ├── dataset.py
    ├── utils.py
    ├── config.py
    ├── inference.py
    ├── evaluate.py
    └── requirements.txt
```

## Approaches

### 1. VAE (Variational Autoencoder)

**Architecture**: Dual encoder-decoder SRVAE with latent space regularization

**Key Features**:
- Encoder-decoder architecture with variational inference
- KL divergence regularization
- MSE + perceptual loss
- Smoother, more stable outputs

**Quick Start**:
```bash
cd VAE
pip install -r requirements.txt
python train.py
```

**Results**:
- Higher PSNR and SSIM metrics
- Smoother reconstructions
- More stable training
- Better pixel-wise accuracy

### 2. GAN (Generative Adversarial Network)

**Architecture**: SRGAN with residual blocks and adversarial training

**Key Features**:
- Generator with 16 residual blocks
- Discriminator for adversarial training
- Combined pixel, perceptual, and adversarial losses
- Photo-realistic texture synthesis

**Quick Start**:
```bash
cd GAN
pip install -r requirements.txt
python train_srgan.py
```

**Results**:
- Sharper, more realistic textures
- Better perceptual quality
- Photo-realistic details
- May produce artifacts

## Comparison

| Metric | VAE (SRVAE) | GAN (SRGAN) |
|--------|-------------|-------------|
| **PSNR** | Higher (~26-28 dB) | Lower (~24-26 dB) |
| **SSIM** | Higher (~0.85-0.90) | Lower (~0.80-0.85) |
| **Training Time** | Faster (~1-2 hours) | Slower (~2-3 hours) |
| **Stability** | Very stable | Less stable |
| **Visual Quality** | Smooth, slightly blurry | Sharp, realistic textures |
| **Artifacts** | Minimal | Occasional |
| **Best For** | Metric-driven tasks | Perceptual realism |

## Dataset

Both implementations use the **CelebA dataset** (aligned & cropped faces):
- High-resolution: 256×256 pixels
- Low-resolution: 64×64 pixels (downsampled)
- Training: 10,000-30,000 images recommended

Download: [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

## Requirements

### VAE Requirements
```bash
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
numpy>=1.23.0
```

### GAN Requirements
```bash
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
matplotlib>=3.5.0
tqdm>=4.65.0
numpy>=1.23.0
```

## Usage

### Training

**VAE**:
```bash
cd VAE
python train.py --data_path /path/to/celeba --epochs 100
```

**GAN**:
```bash
cd GAN
python train_srgan.py
```

### Inference

**VAE**:
```bash
cd VAE
python inference.py --model checkpoint.pth --input image.jpg
```

**GAN**:
```bash
cd GAN
python inference.py --model generator_final.pth --input image.jpg --compare
```

### Evaluation

**GAN** (includes comprehensive metrics):
```bash
cd GAN
python evaluate.py --model generator_final.pth --data /path/to/test/images
```

## Key Differences

### Training Objective
- **VAE**: Minimize reconstruction error + KL divergence
- **GAN**: Adversarial training (generator vs discriminator)

### Loss Functions
- **VAE**: MSE + KL divergence + (optional perceptual)
- **GAN**: Pixel MSE + VGG perceptual + Adversarial

### Output Characteristics
- **VAE**: Smoother, more consistent, higher metrics
- **GAN**: Sharper textures, more realistic, lower metrics

### When to Use Each
- **Use VAE when**: You need stable training, better metrics, or smoother outputs
- **Use GAN when**: You prioritize perceptual quality and photo-realism

## Results Visualization

Sample outputs from both models can be found in their respective output directories:
- `VAE/outputs/` - VAE sample reconstructions
- `GAN/srgan_outputs/samples/` - GAN sample comparisons

## Hardware Requirements

- **GPU**: NVIDIA GPU with CUDA support recommended
- **VRAM**: 8GB+ recommended for batch size 8
- **RAM**: 16GB+ system memory
- **Storage**: 10GB+ for dataset and outputs

Tested on: RTX 5070 (16GB VRAM)


