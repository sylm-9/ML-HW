import os
import glob
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, names, transform):
        self.transform = transform
        self.names = names

    def __getitem__(self, idx):
        name = self.names[idx]
        img = torchvision.io.read_image(name)
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.names)


def get_transform():
    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
    return transform

