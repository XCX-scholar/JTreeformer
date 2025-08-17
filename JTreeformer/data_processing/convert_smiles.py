import json
import os
import sys
import pickle
import argparse
from multiprocessing import Pool
from tqdm import tqdm
from typing import Optional, Tuple
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jtnn_utils.mol_tree import MolTree

MAX_TREE_WIDTH=50

def convert_to_mol_tree(smi: str) -> Tuple[Optional[MolTree], Optional[set]]:
    """
    Converts a single SMILES string to a MolTree object.
    Returns None if the conversion fails.
    """
    clique_set = set()
    try:
        mol_tree = MolTree(smi)
        # These are large and not needed after initialization
        del mol_tree.mol
        for node in mol_tree.nodes:
            del node.mol
            clique_set.add(node.smiles)
        return (mol_tree,clique_set)
    except Exception as e:
        print(f"Failed to process SMILES '{smi}': {e}", file=sys.stderr)
        return (None, None)

def main():
    """
    Main function to run the conversion process.
    """
    parser = argparse.ArgumentParser(
        description="Convert a file of SMILES strings into a list of MolTree objects."
    )
    parser.add_argument(
        '--smiles_file',
        type=str,
        required=True,
        help="Path to the input text file containing one SMILES string per line."
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help="Path to save the output .pkl file containing the list of MolTree objects."
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=os.cpu_count()//2,
        help="Number of parallel processes to use for conversion."
    )
    args = parser.parse_args()
    smiles_file_name = os.path.basename(args.smiles_file)
    dataset_name, _ = smiles_file_name.split('.', 2)

    print(f"Reading SMILES from {args.smiles_file}...")
    with open(args.smiles_file, 'r') as f:
        smiles_list = [line.strip() for line in f if line.strip()]

    print(f"Found {len(smiles_list)} SMILES to process using {args.num_workers} workers.")

    with Pool(args.num_workers) as pool:
        results = list(tqdm(
            pool.imap(convert_to_mol_tree, smiles_list),
            total=len(smiles_list),
            desc="Converting SMILES to MolTrees"
        ))

    mol_trees = [result[0] for result in results if result[0] is not None and result[0]]
    clique_sets = [result[1] for result in results if result[1] is not None and result[1]]

    print(f"Successfully converted {len(mol_trees)} SMILES.")
    print(f"Saving MolTree list to {args.output_path}...")

    with open(os.path.join(args.output_path, f"{dataset_name}_mol_tree.pkl"), 'wb') as f:
        pickle.dump(mol_trees, f)

    vocab = set()
    for cset in clique_sets:
        vocab = vocab | cset
    vocab_map = {v:i for i, v in enumerate(vocab)}
    vocab_map['stop'] = len(vocab)
    with open(os.path.join(args.output_path, f"{dataset_name}_vocab.json"), "a") as file:
        json.dump(vocab_map, file, indent=4)

    print("Conversion complete.")


if __name__ == "__main__":
    main()
