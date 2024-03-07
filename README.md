# JTreeformer

This repository contains code and resources for training Denoising Diffusion Probabilistic Models (DDPM) and Variational Autoencoder (VAE) models. These models are used for generative modeling and probabilistic inference tasks, applicable in image generation, data compression, and other probabilistic inference tasks.

## Contents

- `DDDM_train.py`: Python script for training DDIM model.
- `vae_train.py`: Python script for training VAE model.
- `requirements.txt`: Lists Python dependencies required for the project.

## How to Use

### Clone this repository to your local machine:

```bash
git clone https://github.com/your_username/ddpm-vae-training.git
```

### Training
#### Train the vae model:
First, the Jtreeformer's vae should be trained for latent space generation:
```bash
python vae_train.py --parameters --model_path path/to/save/model
```

#### Train the DDIM model:
The DDIM model is trained for sampling in latent space:
```bash
python DDDM_train.py --parameters --model_path path/to/save/model
```

### Testing
```bash
python Generate3.py --parameters --store_path path/to/save/result
```
