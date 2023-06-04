import torch
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms


class HW8Dataset(TensorDataset):
    def __init__(self, tensors):
        self.tensors = tensors
        if tensors.shape[-1] == 3:
            self.tensors = tensors.permute(0, 3, 1, 2)
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x.to(torch.float32)),
            transforms.Lambda(lambda x: 2. * x / 255. - 1.)
        ])

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, item):
        return self.transform(self.tensors[item])
