import torch
import numpy as np
from torch.utils.data import random_split


def same_seed(seed):
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_valid_split(data_set, valid_ratio, seed):
    valid_size = int(valid_ratio * len(data_set))
    train_set, valid_set = random_split(data_set, [len(data_set)-valid_size, valid_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)


