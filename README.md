# JTreeformer Molecular Generation Pipeline

This document provides instructions on how to set up the environment, prepare the data, and run the complete training and evaluation pipeline for the JTreeformer models (VAE and DDPM).

## 1. Project Structure & Data Setup

Before running the pipeline, ensure your project is structured correctly. The script expects the dataset to be located in a directory relative to the source code folder.

### Directory Layout

The `main.py` script and `run_pipeline.sh`  are in the directory `scripts`. Your data should be in a parallel directory `data`.

```
JTreeformer/
├── data/
│   └── tool_dataset.smi  <--Place your dataset file
├── scripts/
│   ├── main.py
│   ├── run_pipeline.sh
│   └── ... (other modules)
├── utils/
│   └── config.py             <-- Model architecture hyperparameters
└── checkpoints_vae/      <-- Will be created automatically
└── checkpoints_ddpm/     <-- Will be created automatically
└── results/              <-- Create and specify this directory for JSON results, or results will be stored with the checkpoints
```

### Dataset Format

The pipeline processes datasets in the **SMILES** format.
- The file should contain one SMILES string per line.
- The default expected file extension is `.smi`.
- You can specify the dataset name and extension using the `--dataset_name` and `--dataset_suffix` arguments. For `tool_dataset.smi`, you would use `--dataset_name tool_dataset` and `--dataset_suffix smi`.

---

## 3. Configuration: Model Architecture

If you wish to experiment with the model's structure, you can modify the values in that file. This includes parameters such as:
- Latent space dimensionality (`latent_dim`)
- Hidden layer sizes (`hidden_dim`)
- Number of layers, attention heads, etc.

Changes in `utils/config.py` are considered foundational and will affect any new model you train. For reproducibility, it's recommended to track any changes you make to this file.

## 3. How to Run the Pipeline

You can run the full pipeline using either the provided shell script (recommended for ease of use) or by calling the Python script directly with arguments.

### Method 1: Using the Shell Script (Recommended)

The `run_pipeline.sh` script is pre-configured to run the entire process. You can easily customize all parameters by editing the variables at the top of the file.

1.  **Navigate** to the source directory where `main.py` is located.
    ```bash
    cd project_root/JTreeformer
    ```

2.  **Make the script executable** (only needs to be done once).
    ```bash
    chmod +x ./scripts/run_pipeline.sh
    ```

3.  **Execute the script**.
    ```bash
    ./scripts/run_pipeline.sh
    ```

### Method 2 (Recommend): Using the Python Script Directly

You can also call `main.py` directly from the command line. This is useful for running specific parts of the pipeline or for quick tests.

1.  **Navigate** to the source directory.
    ```bash
    cd project_root/JTreeformer
    ```

2.  **Run the script** with your desired arguments. For example, to only train and evaluate the VAE:
    ```bash
    python ./scripts/main.py \
        --train_vae True \
        --evaluate_vae True \
        --train_ddpm False \
        --evaluate_ddpm False \
        --data_dir ../data \
        --dataset_name tool_dataset \
        --vae_epochs 10 \
        --vae_batch_size 32 \
        --device cuda
    ```

---

## 4. Configuration Parameters

All command-line arguments are documented below. You can set these in the `run_pipeline.sh` script or pass them directly to `main.py`.

### General Parameters

| Parameter | Default        | Description                                                                        |
| :--- |:---------------|:-----------------------------------------------------------------------------------|
| `--train_vae` | `True`         | Set to `True` to run the VAE training pipeline.                                    |
| `--evaluate_vae` | `True`         | Set to `True` to run VAE evaluation.                                               |
| `--train_ddpm` | `True`         | Set to `True` to run the DDPM training pipeline.                                   |
| `--evaluate_ddpm` | `True`         | Set to `True` to run DDPM evaluation.                                              |
| `--data_dir` | `../data`      | Path to the directory containing the dataset file.                                 |
| `--dataset_name` | `tool_dataset` | The base name of the SMILES dataset file. 'tool_dataset' is just an example        |
| `--dataset_suffix` | `smi`          | The file extension of the dataset.                                                 |
| `--force_preprocess`| `(flag)`       | If present, forces regeneration of all cached data (MolTrees, vocab, LMDB files).  |
| `--seed` | `3407`         | Random seed for reproducibility.                                                   |
| `--test_size` | `0.1`          | Fraction of the dataset to be used for the test set.                               |
| `--valid_size` | `0.1`          | Fraction of the dataset to be used for the validation set.                         |
| `--predict_properties`| `True`         | Set to `True` to enable molecular property prediction (MW, logP, TPSA) in the VAE. |
| `--device` | `cuda`         | Device to use for training (`cuda` or `cpu`).                                      |

### VAE Hyperparameters

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--vae_checkpoint_dir` | `../checkpoints_vae` | Directory to save VAE checkpoints. |
| `--vae_checkpoint_path`| `None` | Path to a specific VAE checkpoint to load for evaluation or DDPM data generation. |
| `--vae_results_path` | `None` | Path to save VAE evaluation results as a JSON file. |
| `--vae_resume_checkpoint`| `(flag)` | If present, resumes VAE training from the latest checkpoint in `vae_checkpoint_dir`. |
| `--vae_epochs` | `1` | Number of epochs for VAE training. |
| `--vae_batch_size` | `4` | Batch size for VAE training. |
| `--vae_lr` | `1e-4` | Learning rate for the VAE. |
| `--vae_warmup_steps` | `4000` | Number of warmup steps for the VAE learning rate scheduler. |
| `--vae_weight_decay` | `0.01` | Weight decay for the VAE optimizer. |
| `--vae_clip_norm` | `1.0` | Gradient clipping norm for the VAE. |
| `--vae_kl_cycle_len` | `5000` | Length of the KL annealing cycle in training steps. |
| `--vae_log_interval` | `100` | Log training progress every N steps. |

### DDPM Hyperparameters

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `--ddpm_checkpoint_dir`| `../checkpoints_ddpm`| Directory to save DDPM checkpoints. |
| `--ddpm_checkpoint_path`| `None` | Path to a specific DDPM checkpoint to load for evaluation. |
| `--ddpm_results_path` | `None` | Path to save DDPM evaluation results as a JSON file. |
| `--ddpm_resume_checkpoint`| `None` | Path to a specific DDPM checkpoint to resume training from. |
| `--ddpm_epochs` | `1` | Number of epochs for DDPM training. |
| `--ddpm_batch_size` | `128` | Batch size for DDPM training. |
| `--ddpm_lr` | `1e-4` | Learning rate for the DDPM. |
| `--ddpm_warmup_steps`| `500` | Number of warmup steps for the DDPM learning rate scheduler. |
| `--ddpm_weight_decay`| `0.0` | Weight decay for the DDPM optimizer. |
| `--ddpm_loss_type` | `l1` | Loss function for the DDPM (`l1` or `l2`). |
| `--ddpm_log_interval`| `10` | Log training progress every N steps. |
