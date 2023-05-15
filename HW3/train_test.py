import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import FoodDataset
from model import Classifier
import numpy as np

if __name__ == "__main__":
    mode = 'train'
    batch_size = 64
    epochs = 1000
    early_stop = 50
    model_path = './model.ckpt'
    train_dir = "training"
    val_dir = "validation"
    test_dir = "test"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_files = [os.path.join(train_dir, x)for x in os.listdir(train_dir) if x.endswith('.jpg')]
    val_files = [os.path.join(val_dir, x) for x in os.listdir(val_dir) if x.endswith('.jpg')]
    test_files = [os.path.join(test_dir, x) for x in os.listdir(val_dir) if x.endswith('.jpg')]
    train_set = FoodDataset(mode='train', files=train_files)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_set = FoodDataset(mode='test', files=val_files)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_set = FoodDataset(mode='test', files=test_files)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print("data load")
    model = Classifier()
    model.to(device)
    print("model load")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=16, T_mult=2)
    best_acc = 0.0
    early_stop_count = 0
    if mode == 'train':
        print("start train")
        for epoch in range(epochs):
            train_acc = 0.0
            train_loss = 0.0
            val_acc = 0.0
            val_loss = 0.0

            model.train()
            for i, batch in enumerate(train_loader):
                img, label = batch
                optimizer.zero_grad()
                outputs = model(img.to(device))
                loss = criterion(outputs, label.to(device))
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(),max_norm=10)
                optimizer.step()
                correct = (outputs.argmax(dim=-1) == label.to(device)).float().mean()
                train_acc += correct.item()
                train_loss += loss.item()
            train_acc = train_acc/(i+1)
            train_loss = train_loss / (i+1)
            print('Epoch: {}, train_acc: {:.5f}, train_loss: {:.5f}'.format(epoch, train_acc, train_loss))
            scheduler.step()

            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    img, label = batch
                    outputs = model(img.to(device))
                    loss = criterion(outputs, label.to(device))
                    correct = (outputs.argmax(dim=-1) == label.to(device)).float().mean()
                    val_acc += correct.item()
                    val_loss += loss.item()
                val_acc = val_acc/(i+1)
                val_loss = val_loss/(i+1)
                print()
                print(
                    'Epoch: {}, val_acc: {:.5f}, val_loss: {:.5f}'.format(epoch, val_acc, val_loss))
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_path)
                print('saving model with acc {:.5f}'.format(best_acc))
                early_stop_count = 0
            else:
                early_stop_count += 1
                if early_stop_count >= early_stop:
                    print("model not improve, early stop.")
                    break
    elif mode == 'test':
        model.load_state_dict(torch.load(model_path))
        test_acc = 0.0
        model.eval()
        prediction =[]
        with torch.no_grad():
            for i, test_img in enumerate(test_loader):
                outputs = model(test_img.to(device)).cpu().numpy()
                test_label = np.argmax(outputs, axis=1)
                prediction.append(test_label.squeeze().tolist())
            print(prediction)