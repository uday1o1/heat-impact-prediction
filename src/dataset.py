import torch
from torch.utils.data import Dataset
import numpy as np

class SeqDataset(Dataset):
    def __init__(self, df, feature_cols, target_cols, lookback):
        self.X = df[feature_cols].values.astype(np.float32)
        self.y = df[target_cols].values.astype(np.int64)  # shape (N, H)
        self.lookback = lookback
        self.idx = np.arange(lookback, len(df))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        t = self.idx[i]
        x_seq = self.X[t-self.lookback:t]      # (L, F)
        y = self.y[t]                          # (H,)
        return torch.from_numpy(x_seq), torch.from_numpy(y)
