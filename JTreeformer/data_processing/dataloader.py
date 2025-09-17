import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
import torch.nn.functional as F

from data_processing.dataset import TreeDataset, LatentDataset, PredictorDataset

def collate_fn(batch):
    """
    Custom collate function to pad PyG data objects into a dense batch.
    Now includes properties.
    """
    pyg_batch = Batch.from_data_list(batch)
    x_list = [data.x for data in batch]
    padded_x = pad_sequence(x_list, batch_first=True, padding_value=0)

    pyg_batch.x_dense = padded_x
    seq_lengths = torch.tensor([len(data.x) for data in batch], dtype=torch.long, device=padded_x.device)
    max_len = padded_x.size(1)
    range_tensor = torch.arange(max_len, device=padded_x.device).expand(len(batch), -1)
    pyg_batch.padding_mask = range_tensor >= seq_lengths.unsqueeze(1)

    # num_nodes_list = [data.num_nodes for data in batch]
    # num_nodes = torch.tensor(num_nodes_list, device=padded_x.device)
    # max_nodes = padded_x.size(1)
    # range_tensor = torch.arange(max_nodes, device=padded_x.device).expand(len(batch), -1)
    # virtual_token_indices = (num_nodes - 1).unsqueeze(1)
    # pyg_batch.padding_mask_nodes = range_tensor >= virtual_token_indices

    for key in ['hs', 'layer_number', 'degree', 'parent_pos']:
        feat_list = [getattr(data, key) for data in batch]
        setattr(pyg_batch, f'{key}_dense', pad_sequence(feat_list, batch_first=True, padding_value=0))
        # print(key, pyg_batch.__getattribute__(f'{key}_dense').shape)
    pyg_batch.relations_dense = pad_sequence([data.relations for data in batch], batch_first=True, padding_value=-1)

    # Handle properties if they exist
    if 'properties' in batch[0]:
        properties_list = [data.properties for data in batch]
        pyg_batch.properties = torch.stack(properties_list, dim=0)

    # Create dense adjacency matrix
    max_nodes = pyg_batch.x_dense.size(1)
    adj_matrices = []
    for data in batch:
        adj = torch.sparse_coo_tensor(
            data.edge_index, torch.ones(data.edge_index.size(1)), (data.num_nodes, data.num_nodes)
        ).to_dense()
        padded_adj = F.pad(adj, (0, max_nodes - data.num_nodes, 0, max_nodes - data.num_nodes))
        adj_matrices.append(padded_adj)
    pyg_batch.adj_dense = torch.stack(adj_matrices)

    return pyg_batch

def create_vae_dataloader(dataset_path: str, batch_size: int, shuffle: bool = True, num_workers: int = 0):
    """ Creates a DataLoader for the VAE's TreeDataset. """
    dataset = TreeDataset(lmdb_path=dataset_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)

def create_ddpm_dataloader(dataset_path: str, batch_size: int, shuffle: bool = True, num_workers: int = 0):
    """ Creates a DataLoader for the DDPM's LatentDataset. """
    dataset = LatentDataset(latent_path=dataset_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def create_predictor_dataloader(data_path: str, batch_size: int, targets_config: dict, shuffle: bool = True, num_workers: int = 0):
    """ Creates a DataLoader for the Predictor's PredictorDataset. """
    dataset = PredictorDataset(data_path=data_path, targets_config=targets_config)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
