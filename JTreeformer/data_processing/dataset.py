import pickle
import lmdb
from torch.utils.data import Dataset
from torch_geometric.data import Data


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