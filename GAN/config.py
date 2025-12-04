"""
Configuration
Training hyperparameters and settings for SRGAN
"""


class Config:
    """Configuration class for SRGAN training"""

    # Dataset settings
    DATA_PATH = r"C:\Users\satya\OneDrive\Documents\SuperResolution\VAE\data\img_align_celeba"
    HR_SIZE = 256
    LR_SIZE = 64
    MAX_IMAGES = 10000  # None for all images

    # Training settings
    NUM_EPOCHS = 100
    BATCH_SIZE = 8
    NUM_WORKERS = 4

    # Model architecture
    NUM_RESIDUAL_BLOCKS = 16

    # Optimizer settings
    LR_GENERATOR = 0.0001
    LR_DISCRIMINATOR = 0.0001
    BETA1 = 0.9
    BETA2 = 0.999

    # Loss weights
    LAMBDA_PIXEL = 1.0
    LAMBDA_PERCEPTUAL = 0.006
    LAMBDA_ADVERSARIAL = 0.001

    # Learning rate scheduler
    SCHEDULER_STEP = 30
    SCHEDULER_GAMMA = 0.5

    # Checkpoint and logging
    SAVE_DIR = './srgan_outputs'
    CHECKPOINT_DIR = './srgan_outputs/checkpoints'
    SAMPLE_DIR = './srgan_outputs/samples'

    SAVE_SAMPLE_EVERY = 5  # epochs
    SAVE_CHECKPOINT_EVERY = 10  # epochs

    # Device
    DEVICE = 'cuda'  # 'cuda' or 'cpu'

    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 70)
        print("TRAINING CONFIGURATION")
        print("=" * 70)
        print(f"Data Path: {cls.DATA_PATH}")
        print(f"Resolution: {cls.LR_SIZE} -> {cls.HR_SIZE} ({cls.HR_SIZE // cls.LR_SIZE}x)")
        print(f"Max Images: {cls.MAX_IMAGES if cls.MAX_IMAGES else 'All'}")
        print(f"\nTraining:")
        print(f"  Epochs: {cls.NUM_EPOCHS}")
        print(f"  Batch Size: {cls.BATCH_SIZE}")
        print(f"  Learning Rate (G): {cls.LR_GENERATOR}")
        print(f"  Learning Rate (D): {cls.LR_DISCRIMINATOR}")
        print(f"\nLoss Weights:")
        print(f"  Pixel: {cls.LAMBDA_PIXEL}")
        print(f"  Perceptual: {cls.LAMBDA_PERCEPTUAL}")
        print(f"  Adversarial: {cls.LAMBDA_ADVERSARIAL}")
        print(f"\nModel:")
        print(f"  Residual Blocks: {cls.NUM_RESIDUAL_BLOCKS}")
        print(f"\nOutput:")
        print(f"  Save Directory: {cls.SAVE_DIR}")
        print("=" * 70)


# Preset configurations for different training scenarios

class FastConfig(Config):
    """Fast training configuration for testing"""
    MAX_IMAGES = 5000
    NUM_EPOCHS = 50
    BATCH_SIZE = 16


class StandardConfig(Config):
    """Standard configuration for good results"""
    MAX_IMAGES = 10000
    NUM_EPOCHS = 100
    BATCH_SIZE = 8


class HighQualityConfig(Config):
    """High-quality configuration for best results"""
    MAX_IMAGES = 30000
    NUM_EPOCHS = 150
    BATCH_SIZE = 8
    NUM_RESIDUAL_BLOCKS = 23