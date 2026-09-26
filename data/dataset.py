import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os


class TimeSeriesDataset(Dataset):
    """
    Loads preprocessed .npy files for a given dataset and split.
    Returns (data, mask) tensors of shape (n_features, seq_len).
    mask is 1 where a feature was originally observed, 0 where imputed.
    Falls back to all-ones mask if no mask file exists.
    """

    def __init__(self, data_dir, dataset_name, split='train'):
        path = os.path.join(data_dir, f'{dataset_name}_{split}.npy')
        mask_path = os.path.join(data_dir, f'{dataset_name}_{split}_mask.npy')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Processed data not found at {path}. Run the loader script first.")
        data = np.load(path)                   # (N, seq_len, n_features)
        self.data = torch.tensor(data, dtype=torch.float32).permute(0, 2, 1)  # (N, n_features, seq_len)

        if os.path.exists(mask_path):
            mask = np.load(mask_path)          # (N, seq_len, n_features)
            self.mask = torch.tensor(mask, dtype=torch.float32).permute(0, 2, 1)
        else:
            self.mask = torch.ones_like(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.mask[idx]  # each (n_features, seq_len)


def get_dataloader(data_dir, dataset_name, split, batch_size, num_workers=2, shuffle=None):
    if shuffle is None:
        shuffle = (split == 'train')
    dataset = TimeSeriesDataset(data_dir, dataset_name, split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=(split == 'train'))
