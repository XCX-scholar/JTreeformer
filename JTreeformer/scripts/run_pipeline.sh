#!/bin/bash

# ==============================================================================
# JTreeformer Full Pipeline Execution Script
# ==============================================================================

# --- CONFIGURATION ---
# Set the project's source root directory.
# This assumes the script is run from the directory containing `main.py`.
SOURCE_ROOT=$(pwd)

# -- General Arguments --
TRAIN_VAE=True
EVALUATE_VAE=True
TRAIN_DDPM=True
EVALUATE_DDPM=True
DATA_DIR="./data"
DATASET_NAME="tool_dataset"
DATASET_SUFFIX="smi"
FORCE_PREPROCESS=false # Use 'true' to force regeneration of all data
SEED=3407
TEST_SIZE=0.1
VALID_SIZE=0.1
PREDICT_PROPERTIES=True
DEVICE="cuda" # "cuda" or "cpu"

# -- VAE Hyperparameters --
VAE_CHECKPOINT_DIR="../checkpoints_vae"
VAE_CHECKPOINT_PATH=""
VAE_RESULTS_PATH=""
VAE_RESUME_CHECKPOINT=false
VAE_EPOCHS=10
VAE_BATCH_SIZE=32
VAE_LR=1e-4
VAE_WARMUP_STEPS=4000
VAE_WEIGHT_DECAY=0.01
VAE_CLIP_NORM=1.0
VAE_KL_CYCLE_LEN=5000
VAE_LOG_INTERVAL=100

# -- DDPM Hyperparameters --
DDPM_CHECKPOINT_DIR="../checkpoints_ddpm"
DDPM_CHECKPOINT_PATH=""
DDPM_RESULTS_PATH=""
DDPM_RESUME_CHECKPOINT=""
DDPM_EPOCHS=20
DDPM_BATCH_SIZE=128
DDPM_LR=1e-4
DDPM_WARMUP_STEPS=500
DDPM_WEIGHT_DECAY=0.0
DDPM_LOSS_TYPE="l1" # "l1" or "l2"
DDPM_LOG_INTERVAL=10

set -e

# Set PYTHONPATH to include the source root and source_root/scripts
export PYTHONPATH=$PYTHONPATH:$SOURCE_ROOT:$SOURCE_ROOT/scripts
echo "PYTHONPATH set to: $PYTHONPATH"

CMD_ARGS=""

# -- General arguments --
CMD_ARGS+=" --train_vae ${TRAIN_VAE}"
CMD_ARGS+=" --evaluate_vae ${EVALUATE_VAE}"
CMD_ARGS+=" --train_ddpm ${TRAIN_DDPM}"
CMD_ARGS+=" --evaluate_ddpm ${EVALUATE_DDPM}"
CMD_ARGS+=" --data_dir ${DATA_DIR}"
CMD_ARGS+=" --dataset_name ${DATASET_NAME}"
CMD_ARGS+=" --dataset_suffix ${DATASET_SUFFIX}"
CMD_ARGS+=" --seed ${SEED}"
CMD_ARGS+=" --test_size ${TEST_SIZE}"
CMD_ARGS+=" --valid_size ${VALID_SIZE}"
CMD_ARGS+=" --predict_properties ${PREDICT_PROPERTIES}"
CMD_ARGS+=" --device ${DEVICE}"

if [ "$FORCE_PREPROCESS" = true ]; then
    CMD_ARGS+=" --force_preprocess"
fi

# -- VAE arguments --
CMD_ARGS+=" --vae_checkpoint_dir ${VAE_CHECKPOINT_DIR}"
if [ -n "$VAE_CHECKPOINT_PATH" ]; then
    CMD_ARGS+=" --vae_checkpoint_path ${VAE_CHECKPOINT_PATH}"
fi
if [ -n "$VAE_RESULTS_PATH" ]; then
    CMD_ARGS+=" --vae_results_path ${VAE_RESULTS_PATH}"
fi
if [ "$VAE_RESUME_CHECKPOINT" = true ]; then
    CMD_ARGS+=" --vae_resume_checkpoint"
fi
CMD_ARGS+=" --vae_epochs ${VAE_EPOCHS}"
CMD_ARGS+=" --vae_batch_size ${VAE_BATCH_SIZE}"
CMD_ARGS+=" --vae_lr ${VAE_LR}"
CMD_ARGS+=" --vae_warmup_steps ${VAE_WARMUP_STEPS}"
CMD_ARGS+=" --vae_weight_decay ${VAE_WEIGHT_DECAY}"
CMD_ARGS+=" --vae_clip_norm ${VAE_CLIP_NORM}"
CMD_ARGS+=" --vae_kl_cycle_len ${VAE_KL_CYCLE_LEN}"
CMD_ARGS+=" --vae_log_interval ${VAE_LOG_INTERVAL}"

# -- DDPM arguments --
CMD_ARGS+=" --ddpm_checkpoint_dir ${DDPM_CHECKPOINT_DIR}"
if [ -n "$DDPM_CHECKPOINT_PATH" ]; then
    CMD_ARGS+=" --ddpm_checkpoint_path ${DDPM_CHECKPOINT_PATH}"
fi
if [ -n "$DDPM_RESULTS_PATH" ]; then
    CMD_ARGS+=" --ddpm_results_path ${DDPM_RESULTS_PATH}"
fi
if [ -n "$DDPM_RESUME_CHECKPOINT" ]; then
    CMD_ARGS+=" --ddpm_resume_checkpoint ${DDPM_RESUME_CHECKPOINT}"
fi
CMD_ARGS+=" --ddpm_epochs ${DDPM_EPOCHS}"
CMD_ARGS+=" --ddpm_batch_size ${DDPM_BATCH_SIZE}"
CMD_ARGS+=" --ddpm_lr ${DDPM_LR}"
CMD_ARGS+=" --ddpm_warmup_steps ${DDPM_WARMUP_STEPS}"
CMD_ARGS+=" --ddpm_weight_decay ${DDPM_WEIGHT_DECAY}"
CMD_ARGS+=" --ddpm_loss_type ${DDPM_LOSS_TYPE}"
CMD_ARGS+=" --ddpm_log_interval ${DDPM_LOG_INTERVAL}"

echo "Running main.py with the following arguments:"
echo "${SOURCE_ROOT}/scripts/main.py ${CMD_ARGS}"
echo "-------------------------------------------------"

python ${SOURCE_ROOT}/scripts/main.py ${CMD_ARGS}

echo "-------------------------------------------------"
echo "Pipeline finished."
