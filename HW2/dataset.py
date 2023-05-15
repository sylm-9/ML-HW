import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, x, y=None):
        self.data = x
        if y is not None:
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)