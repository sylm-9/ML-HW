import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler

from dataset import HW8Dataset
from model import ResNet

if __name__ == "__main__":
    epochs = 200
    batch_size = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = np.load("./data/trainingset.npy", allow_pickle=True)
    test_data = np.load("./data/testingset.npy", allow_pickle=True)
    print(train_data.shape)
    print(test_data.shape)
    train_dataset = HW8Dataset(torch.from_numpy(train_data))
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)
    model = ResNet()
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(),lr =1e-4)
    best_loss = np.inf
    # training
    for epoch in range(epochs):
        total_loss = list()
        model.train()
        for data in train_dataloader:
            img = data.float().to(device)
            out = model(img)
            loss = criterion(out, img)
            total_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        mean_loss = np.mean(total_loss)
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model, "best_model.pt")

        print('Epoch: {}, Train_Loss: {:.5f}'.format(epoch, mean_loss))


