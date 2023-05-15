import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class FoodDataset(Dataset):
    def __init__(self,  mode='test', files=None):
        super(FoodDataset).__init__()
        self.files = files
        self.transform = get_transform(mode)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        img = Image.open(file_name)
        img = self.transform(img)
        try:
            label = int(file_name.split("/")[-1].split("_")[0])
        except:
            label = -1
        return img, label


def get_transform(mode):
    if mode == 'test':
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
    elif mode == 'train':
        transform =  transforms.Compose([
            transforms.RandomResizedCrop((128, 128), scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(180),
            transforms.RandomAffine(30),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ])
    return transform