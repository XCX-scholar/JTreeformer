import os
import sys
import pickle
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from typing import List
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from jtnn_utils.mol_tree import MolTree
from data_processing.convert_smiles import convert_to_mol_tree
from data_processing.preprocess_data import preprocess_and_save, generate_latent_dataset
from scripts.training_vae import Trainer as VAE_Trainer
from scripts.training_ddpm import Trainer as DDPM_Trainer
from utils.config import VAEConfig, DDPMConfig


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_and_save_scaler(mol_trees_path: str, scaler_path: str):
    print(f"Loading MolTrees from {mol_trees_path} to compute property scaler...")
    with open(mol_trees_path, 'rb') as f:
        mol_trees: List[MolTree] = pickle.load(f)
    properties = np.array([[t.w, t.logp, t.tpsa] for t in mol_trees])
    scaler_data = {'mean': properties.mean(axis=0).tolist(), 'std': properties.std(axis=0).tolist()}
    with open(scaler_path, 'w') as f:
        json.dump(scaler_data, f, indent=2)
    print(f"Property scaler saved to {scaler_path}")


def main():
    parser = argparse.ArgumentParser(description="JTreeformer Full Pipeline")

    general_group = parser.add_argument_group('General')
    general_group.add_argument('--train_vae', action='store_true', help="Run the VAE training pipeline.")
    general_group.add_argument('--evaluate_vae', action='store_true', help="Run VAE evaluation.")
    general_group.add_argument('--train_ddpm', action='store_true', help="Run the DDPM training pipeline.")
    general_group.add_argument('--evaluate_ddpm', action='store_true', help="Run DDPM evaluation.")
    general_group.add_argument('--data_dir', type=str, default='../data/processed/')
    general_group.add_argument('--dataset_name', default='tool_dataset', type=str, help="Name of SMILES dataset file.")
    general_group.add_argument('--dataset_suffix', default='smi', type=str, help="Suffix of dataset file.")
    general_group.add_argument('--force_preprocess', action='store_true', help="Force regeneration of all preprocessed data.")
    general_group.add_argument('--seed', type=int, default=3407)
    general_group.add_argument('--test_size', type=float, default=0.1)
    general_group.add_argument('--valid_size', type=float, default=0.1)
    general_group.add_argument('--predict_properties', type=bool, default=True, help="Enable property prediction in VAE.")
    general_group.add_argument('--device', type=str, default='cuda', help="Device to use ('cuda' or 'cpu').")

    # --- VAE specific arguments ---
    vae_group = parser.add_argument_group('VAE Hyperparameters')
    vae_group.add_argument('--vae_checkpoint_dir', type=str, default='../checkpoints_vae')
    vae_group.add_argument('--vae_checkpoint_path', type=str, default=None, help="Specific VAE checkpoint to load for evaluation or DDPM data generation.")
    vae_group.add_argument('--vae_resume_checkpoint', action='store_true', default=False)
    vae_group.add_argument('--vae_epochs', type=int, default=4)
    vae_group.add_argument('--vae_batch_size', type=int, default=4)
    vae_group.add_argument('--vae_lr', type=float, default=1e-4)
    vae_group.add_argument('--vae_warmup_steps', type=int, default=4000)
    vae_group.add_argument('--vae_weight_decay', type=float, default=0.01)
    vae_group.add_argument('--vae_clip_norm', type=float, default=1.0)
    vae_group.add_argument('--vae_kl_cycle_len', type=int, default=5000)
    vae_group.add_argument('--vae_log_interval', type=int, default=100)

    # --- DDPM specific arguments ---
    ddpm_group = parser.add_argument_group('DDPM Hyperparameters')
    ddpm_group.add_argument('--ddpm_checkpoint_dir', type=str, default='../checkpoints_ddpm')
    ddpm_group.add_argument('--ddpm_checkpoint_path', type=str, default=None, help="Specific DDPM checkpoint to load for evaluation.")
    ddpm_group.add_argument('--ddpm_results_path', type=str, default=None, help="Path to save DDPM evaluation results JSON.")
    ddpm_group.add_argument('--ddpm_resume_checkpoint', type=str, default=None, help="Path to DDPM checkpoint to resume training from.")
    ddpm_group.add_argument('--ddpm_epochs', type=int, default=5000)
    ddpm_group.add_argument('--ddpm_batch_size', type=int, default=128)
    ddpm_group.add_argument('--ddpm_lr', type=float, default=1e-4)
    ddpm_group.add_argument('--ddpm_warmup_steps', type=int, default=500)
    ddpm_group.add_argument('--ddpm_weight_decay', type=float, default=0.0)
    ddpm_group.add_argument('--ddpm_loss_type', type=str, default='l1', choices=['l1', 'l2'])
    ddpm_group.add_argument('--ddpm_log_interval', type=int, default=10)

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.data_dir, exist_ok=True)

    # --- Data Preparation Pipeline ---
    dataset_smiles = os.path.join(args.data_dir, f'{args.dataset_name}.{args.dataset_suffix}')
    train_smiles = os.path.join(args.data_dir, f'{args.dataset_name}_train.{args.dataset_suffix}')
    valid_smiles = os.path.join(args.data_dir, f'{args.dataset_name}_valid.{args.dataset_suffix}')
    test_smiles = os.path.join(args.data_dir, f'{args.dataset_name}_test.{args.dataset_suffix}')
    vocab_path = os.path.join(args.data_dir, f'{args.dataset_name}_vocab.json')
    args.vocab_path = vocab_path

    smiles_sources = {'all': dataset_smiles, 'train': train_smiles, 'valid': valid_smiles, 'test': test_smiles}
    mol_tree_paths = {split: os.path.join(args.data_dir, f'{split}_mol_trees.pkl') for split in smiles_sources}
    lmdb_paths = {split: os.path.join(args.data_dir, f'{split}.lmdb') for split in smiles_sources}
    split_indices_path = os.path.join(args.data_dir, f'{args.dataset_name}_split_indices.json')
    scaler_path = os.path.join(args.data_dir, f'{args.dataset_name}_scaler.json')

    # --- STEP 1: Ensure MolTree files exist for specified splits ---
    all_clique_sets = []
    for split, path in smiles_sources.items():
        if path and os.path.exists(path) and (not os.path.exists(mol_tree_paths[split]) or args.force_preprocess):
            print(f"--- Converting {split} SMILES to MolTrees ---")
            with open(path, 'r') as f:
                smiles_list = [line.strip() for line in f if line.strip()]
            with Pool(os.cpu_count()) as pool:
                results = list(tqdm(pool.imap(convert_to_mol_tree, smiles_list), total=len(smiles_list)))
            mol_trees = [result[0] for result in results if result[0] is not None and result[0]]
            clique_sets = [result[1] for result in results if result[1] is not None and result[1]]
            all_clique_sets.extend(clique_sets)
            with open(mol_tree_paths[split], 'wb') as f:
                pickle.dump(mol_trees, f)
    if (not os.path.exists(vocab_path) or args.force_preprocess) and all_clique_sets:
        print("--- Generating Vocabulary ---")
        vocab = set.union(*all_clique_sets)
        vocab_map = {v: i for i, v in enumerate(vocab)}
        vocab_map['stop'] = len(vocab)
        with open(vocab_path, "w") as file:
            json.dump(vocab_map, file, indent=4)

    # --- STEP 2: Handle splitting if valid/test sets are not provided ---
    if not all(os.path.exists(mol_tree_paths[s]) for s in ['train', 'valid', 'test']):
        print("A split is missing. Attempting to split from the 'all' dataset.")
        if not os.path.exists(mol_tree_paths['all']):
            raise FileNotFoundError(
                f"Cannot create splits because the source MolTree file is missing: {mol_tree_paths['all']}")

        if os.path.exists(split_indices_path) and not args.force_preprocess:
            print(f"Loading existing splits from {split_indices_path}")
            with open(split_indices_path, 'r') as f:
                split_indices = json.load(f)
        else:
            print("Creating new train/valid/test splits...")
            with open(mol_tree_paths['all'], 'rb') as f:
                all_trees_len = len(pickle.load(f))
            indices = np.arange(all_trees_len)
            train_val_idx, test_idx = train_test_split(indices, test_size=args.test_size, random_state=args.seed)
            train_idx, valid_idx = train_test_split(train_val_idx, test_size=args.valid_size / (1 - args.test_size),
                                                    random_state=args.seed)
            split_indices = {'train': train_idx.tolist(), 'valid': valid_idx.tolist(), 'test': test_idx.tolist()}
            with open(split_indices_path, 'w') as f:
                json.dump(split_indices, f)

        with open(mol_tree_paths['all'], 'rb') as f:
            all_trees = pickle.load(f)
        for split, idx in split_indices.items():
            split_trees = [all_trees[i] for i in idx]
            with open(mol_tree_paths[split], 'wb') as f:
                pickle.dump(split_trees, f)
        print("Successfully created and saved MolTree files for all splits.")

    # --- STEP 3: Preprocess MolTrees to LMDB ---
    if args.predict_properties:
        compute_and_save_scaler(mol_tree_paths['train'], scaler_path)

    for split in ['train', 'valid', 'test']:
        if not os.path.exists(lmdb_paths[split]) or args.force_preprocess:
            if not os.path.exists(mol_tree_paths[split]):
                print(f"Warning: Cannot preprocess {split} set, {mol_tree_paths[split]} not found. Skipping.")
                continue
            print(f"--- Preprocessing {split} MolTrees to LMDB ---")
            with open(mol_tree_paths[split], 'rb') as f:
                mol_trees = pickle.load(f)
            preprocess_and_save(
                mol_trees=mol_trees, vocab_path=vocab_path,
                output_lmdb_path=lmdb_paths[split],
                predict_properties=args.predict_properties
            )

    # --- STEP 4.1: VAE Training and/or Evaluation ---
    if args.train_vae or args.evaluate_vae:
        print("\n--- VAE Pipeline ---")
        vae_config = VAEConfig(predict_properties=args.predict_properties)
        vae_args_ns = argparse.Namespace(
            train_path=lmdb_paths['train'], valid_path=lmdb_paths['valid'], test_path=lmdb_paths['test'],
            scaler_path=scaler_path, vocab_path=vocab_path, checkpoint_dir=args.vae_checkpoint_dir,
            resume_checkpoint=args.vae_resume_checkpoint, epochs=args.vae_epochs, batch_size=args.vae_batch_size,
            lr=args.vae_lr, warmup_steps=args.vae_warmup_steps, weight_decay=args.vae_weight_decay,
            clip_norm=args.vae_clip_norm, predict_properties=args.predict_properties,
            kl_cycle_len=args.vae_kl_cycle_len, log_interval=args.vae_log_interval
        )
        vae_trainer = VAE_Trainer(vae_config, vae_args_ns)

        if args.train_vae:
            vae_trainer.train()

        if args.evaluate_vae:
            ckpt = args.vae_checkpoint_path or os.path.join(args.vae_checkpoint_dir, 'checkpoint_best.pth')
            if not os.path.exists(ckpt): raise FileNotFoundError(f"VAE checkpoint not found at {ckpt}")
            vae_trainer._load_checkpoint(ckpt)
            vae_trainer.evaluate()

        # --- STEP 4.2: DDPM Training / Evaluation ---
        ddpm_latent_paths = {s: os.path.join(args.data_dir, f'{s}_latents.pt') for s in ['train', 'valid', 'test']}

        if args.train_ddpm:
            print("\n--- Preparing Latent Dataset for DDPM ---")
            for split in ['train', 'valid', 'test']:
                if not os.path.exists(ddpm_latent_paths[split]) or args.force_preprocess:
                    print(f"Latent data for '{split}' not found. Generating...")
                    vae_ckpt = args.vae_checkpoint_path or os.path.join(args.vae_checkpoint_dir, 'checkpoint_best.pth')
                    if not os.path.exists(vae_ckpt):
                        raise FileNotFoundError(f"Cannot generate latents. VAE checkpoint not found: {vae_ckpt}")
                    if not os.path.exists(lmdb_paths[split]):
                        raise FileNotFoundError(f"Cannot generate latents. VAE data not found: {lmdb_paths[split]}")

                    generate_latent_dataset(
                        vae_checkpoint_path=vae_ckpt, vae_data_path=lmdb_paths[split],
                        output_path=ddpm_latent_paths[split], vocab_path=vocab_path,
                        batch_size=args.vae_batch_size, device=args.device
                    )

        if args.train_ddpm or args.evaluate_ddpm:
            print("\n--- DDPM Pipeline ---")
            ddpm_config = DDPMConfig()
            ddpm_args_ns = argparse.Namespace(
                train_path=ddpm_latent_paths['train'], valid_path=ddpm_latent_paths['valid'], test_path=ddpm_latent_paths['test'],
                checkpoint_dir=args.ddpm_checkpoint_dir,
                resume_checkpoint=args.ddpm_resume_checkpoint,
                checkpoint_path=args.ddpm_checkpoint_path or os.path.join(args.ddpm_checkpoint_dir,'checkpoint_best.pth'),
                latent_dim=ddpm_config.latent_dim, timesteps=ddpm_config.timesteps,
                loss_type=args.ddpm_loss_type, epochs=args.ddpm_epochs, batch_size=args.ddpm_batch_size,
                lr=args.ddpm_lr, warmup_steps=args.ddpm_warmup_steps, weight_decay=args.ddpm_weight_decay,
                log_interval=args.ddpm_log_interval, device=args.device,
                train=args.train_ddpm, evaluate=args.evaluate_ddpm, results_path=args.ddpm_results_path
            )

            ddpm_trainer = DDPM_Trainer(ddpm_args_ns)

            if args.train_ddpm:
                ddpm_trainer.train()

            if args.evaluate_ddpm:
                ddpm_trainer.evaluate()

        print("\nPipeline finished.")


if __name__ == "__main__":
    main()
