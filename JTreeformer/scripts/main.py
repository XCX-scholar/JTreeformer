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
from data_processing.preprocess_data import preprocess_and_save
from scripts.training import Trainer
from utils.config import ModelConfig


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
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--evaluate', type=bool, default=True)

    parser.add_argument('--data_dir', type=str, default='../data/processed/')
    parser.add_argument('--dataset_name', default='tool_dataset', type=str, help="Name of SMILES dataset file.")
    parser.add_argument('--dataset_suffix', default='smi', type=str, help="Suffix of dataset file.")

    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--force_preprocess', action='store_true')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--valid_size', type=float, default=0.1)
    parser.add_argument('--resume_checkpoint', type=bool, default=False)

    trainer_group = parser.add_argument_group('Training Hyperparameters')
    trainer_group.add_argument('--epochs', type=int, default=4)
    trainer_group.add_argument('--batch_size', type=int, default=4)
    trainer_group.add_argument('--lr', type=float, default=1e-4)
    trainer_group.add_argument('--warmup_steps', type=int, default=4000)
    trainer_group.add_argument('--weight_decay', type=float, default=0.01)
    trainer_group.add_argument('--clip_norm', type=float, default=1.0)
    trainer_group.add_argument('--predict_properties', type=bool, default=True)
    trainer_group.add_argument('--kl_cycle_len', type=int, default=5000)
    trainer_group.add_argument('--log_interval', type=int, default=100)

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
    if not os.path.exists(vocab_path) or args.force_preprocess:
        vocab = set()
        for cset in all_clique_sets:
            vocab = vocab | cset
        vocab_map = {v: i for i, v in enumerate(vocab)}
        vocab_map['stop'] = len(vocab)
        with open(vocab_path, "w") as file:
            json.dump(vocab_map, file, indent=4)

    # --- STEP 2: Handle splitting if valid/test sets are not provided ---
    if not os.path.exists(mol_tree_paths['train']) or not os.path.exists(mol_tree_paths['valid']) or not os.path.exists(mol_tree_paths['test']):
        print("Train or Validation or Test MolTree file not found. Attempting to split from training set.")
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

    # --- STEP 4: Training and/or Evaluation ---
    model_config = ModelConfig()
    args.train_path = lmdb_paths['train']
    args.valid_path = lmdb_paths['valid']
    args.test_path = lmdb_paths['test']
    args.scaler_path = scaler_path

    trainer = Trainer(model_config, args)

    if args.train:
        print("--- Running Training ---")
        trainer.train()

    if args.evaluate:
        print("--- Running Evaluation on Test Set ---")
        checkpoint_to_load = args.checkpoint_path or os.path.join(args.checkpoint_dir, 'checkpoint_best.pth')
        trainer._load_checkpoint(checkpoint_to_load)
        test_losses = trainer._run_epoch(epoch=0, is_train=False, use_test_set=True)
        print("\n--- Test Set Evaluation Results ---")
        print(" | ".join([f"{k}: {v:.4f}" for k, v in test_losses.items()]))
        print("-----------------------------------\n")

    print("Pipeline finished.")


if __name__ == "__main__":
    main()