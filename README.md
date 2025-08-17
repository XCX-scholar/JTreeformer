# JTreeformer

This repository contains code and resources for training Denoising Diffusion Probabilistic Models (DDPM) and Variational Autoencoder (VAE) models, where VAE is the proposed method we utilize for molecule representation in latent space and molecule generation. DDPM is utilized to sample in the latent space.

## Contents

- `main.py`: Python script for training VAE model. DDPM model will be added later.
- `requirements.txt`: Lists Python dependencies required for the project.

## How to Use

### Clone this repository to your local machine:

```bash
git clone https://github.com/XCX-scholar/JTreeformer.git
```

### Training
#### Train the vae model:
First, the Jtreeformer's vae should be trained for latent space generation, modify the utils/config.py to define the model, and run:
```bash
python scripts/main.py --your_training_parameters
```

#### Train the DDIM model:
The DDIM model will be added later

### Testing
Test scripts will be added later.
