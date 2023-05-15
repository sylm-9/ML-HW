import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = torch.FloatTensor(x)
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)


def select_feat(train_data, valid_data, test_data):
    y_train, y_valid = train_data[:, -1], valid_data[:, -1]
    x_train, x_valid, x_test = train_data[:, :-1], valid_data[:, :-1], test_data
    feat_idx = [53, 69, 85, 101]

    return x_train[:, feat_idx], x_valid[:, feat_idx], x_test[:, feat_idx], y_train, y_valid
