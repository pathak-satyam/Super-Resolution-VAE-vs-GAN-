\# Super-Resolution Comparison: VAE vs GAN



Face image super-resolution using deep learning approaches.



\## Project Structure

```

super-resolution-comparison/

├── VAE/              # Variational Autoencoder approach

│   ├── model.py

│   ├── dataset.py

│   ├── loss.py

│   ├── train.py

│   └── requirements.txt

└── GAN/              # Generative Adversarial Network (coming soon)

```



\## VAE Implementation



\- \*\*Model\*\*: Dual encoder-decoder SRVAE architecture

\- \*\*Task\*\*: 64×64 → 256×256 face super-resolution

\- \*\*Dataset\*\*: CelebA



\## Quick Start

```bash

cd VAE

pip install -r requirements.txt

python train.py

```

