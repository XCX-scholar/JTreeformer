import pickle
import lmdb
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch

class TreeDataset(Dataset):
    """
    A PyTorch Dataset for loading preprocessed MolTree data from an LMDB database.

    Each item in the dataset is a torch_geometric.data.Data object containing
    all necessary information for a single tree graph.
    """

    def __init__(self, lmdb_path: str):
        """
        Args:
            lmdb_path (str): Path to the LMDB database directory.
        """
        self.lmdb_path = lmdb_path
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

        with self.env.begin(write=False) as txn:
            self._keys = [key for key, _ in txn.cursor()]
            self.num_samples = len(self._keys)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Data:
        """
        Retrieves a data sample from the LMDB database.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            A torch_geometric.data.Data object.
        """
        key = self._keys[idx]
        with self.env.begin(write=False) as txn:
            pickled_data = txn.get(key)

        data_object = pickle.loads(pickled_data)
        return data_object

class LatentDataset(Dataset):
    """
    A PyTorch Dataset for loading latent vectors from a .pt file, for the DDPM.
    """
    def __init__(self, latent_path: str):
        """
        Args:
            latent_path (str): Path to the .pt file containing the tensor of latent vectors.
        """
        print(f"Loading latent vectors from: {latent_path}")
        self.latents = torch.load(latent_path)

    def __len__(self) -> int:
        return len(self.latents)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Retrieves a latent vector.
        """
        return self.latents[idx]
