import os
import sys
import pickle
import json
import lmdb
from tqdm import tqdm
import torch
from torch_geometric.data import Data
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_processing.serialization import dfs_serialize_tree
from jtnn_utils.mol_tree import MolTree

STOP_TOKEN = "stop"

def preprocess_and_save(
        mol_trees: List[MolTree],
        vocab_path: str,
        output_lmdb_path: str,
        predict_properties: bool
):
    """
    Main function to load raw MolTrees, preprocess them, and save to LMDB.
    Now includes properties if the flag is set.
    """
    with open(vocab_path, 'r') as f:
        vocab_map: dict[str, int] = json.load(f)

    if STOP_TOKEN not in vocab_map:
        vocab_map[STOP_TOKEN] = len(vocab_map)

    map_size = int(1024 * 1024 * 1024 //4)  # 0.25 GB
    env = lmdb.open(output_lmdb_path, map_size=map_size)

    with env.begin(write=True) as txn:
        count = 0
        for i, tree in enumerate(tqdm(mol_trees, desc=f"Preprocessing to {os.path.basename(output_lmdb_path)}")):
            if not tree.nodes: continue

            nodes_in_order, relations, layer_numbers, parent_positions = dfs_serialize_tree(tree)
            if not nodes_in_order: continue

            node_smiles = [node.smiles for node in nodes_in_order]
            node_indices = torch.tensor(
                [vocab_map.get(s, 0) for s in node_smiles] + [vocab_map[STOP_TOKEN]],
                dtype=torch.long
            )

            node_to_idx = {node: i for i, node in enumerate(nodes_in_order)}

            sources, targets = [], []
            for node, idx in node_to_idx.items():
                for neighbor in node.neighbors:
                    if node_to_idx.get(neighbor, -1) > idx:
                        sources.append(idx)
                        targets.append(node_to_idx[neighbor])

            edge_index = torch.tensor([sources, targets], dtype=torch.long)

            data_args = {
                "x": node_indices,
                "edge_index": edge_index,
                "hs": torch.tensor([node.hs for node in nodes_in_order] + [0], dtype=torch.long),
                "layer_number": torch.tensor(layer_numbers + [0], dtype=torch.long),
                "degree": torch.tensor([len(node.neighbors) for node in nodes_in_order] + [0], dtype=torch.long),
                "parent_pos": torch.tensor(parent_positions  + [0], dtype=torch.long),
                "relations": torch.tensor(relations + [0], dtype=torch.long),
            }

            if predict_properties:
                data_args["properties"] = torch.tensor([tree.w, tree.logp, tree.tpsa], dtype=torch.float)

            data = Data(**data_args)

            key = f"{i:08}".encode("ascii")
            txn.put(key, pickle.dumps(data))
            count += 1

    print(f"Successfully processed and saved {count} trees to {output_lmdb_path}")
