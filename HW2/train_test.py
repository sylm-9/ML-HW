import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Classifier
from dataset import MyDataset
import numpy as np
from process_data import preprocess_data
import gc
import torch.nn as nn

if __name__ == "__main__":
    batch_size = 2048
    epochs = 1000
    early_stop = 10
    model_path = './model.ckpt'
    input_dim = 39 * 19
    hidden_layers = 5
    hidden_dim = 1024
    mode = "train"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_X, train_y, val_X, val_y, test_X = preprocess_data()
    train_set = MyDataset(train_X, train_y)
    val_set = MyDataset(val_X, val_y)
    test_set = MyDataset(test_X, None)

    del train_X, train_y, val_X, val_y, test_X
    gc.collect()

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=16, T_mult=2, eta_min=1e-5)

    best_acc = 0.0
    early_stop_count = 0
    if mode == 'train':
        for epoch in range(epochs):
            train_acc = 0.0
            train_loss = 0.0
            val_acc = 0.0
            val_loss = 0.0

            model.train()
            samples = 0
            for i, batch in enumerate(train_loader):
                X, y = batch
                X = X.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                _, prediction = torch.max(outputs, 1)
                correct = (prediction.detach() == y.detach()).sum().item()
                train_acc += correct
                samples += y.size(0)
                train_loss += loss.item()
            print('Epoch: {}, train_acc: {:.5f}, train_loss: {:.5f}'.format(epoch, train_acc / samples, train_loss / (i+1)))
            scheduler.step()
            if len(val_set) > 0:
                model.eval()
                with torch.no_grad():
                    samples = 0
                    for i, batch in enumerate(val_loader):
                        X, y = batch
                        X = X.to(device)
                        y = y.to(device)

                        outputs = model(X)
                        loss = criterion(outputs, y)
                        _, val_prediction = torch.max(outputs, 1)
                        val_acc += (val_prediction.cpu() == y.cpu()).sum().item()
                        samples += y.size(0)
                        val_loss += loss.item()
                    print()
                    print('Epoch: {}, val_acc: {:.5f}, val_loss: {:.5f}'.format(epoch, val_acc / samples, val_loss / (i+1)))
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), model_path)
                    print('saving model with acc {:.5f}'.format(best_acc / len(val_set)))
                    early_stop_count = 0
                else:
                    early_stop_count += 1
                    if early_stop_count >= early_stop:
                        print("model not improve, early stop.")
                        break
    elif mode == 'test':
        model.load_state_dict(torch.load(model_path))
        test_acc = 0.0
        prediction = np.array([], dtype=np.int32)

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                test_X = batch
                test_X = test_X.to(device)

                outputs = model(test_X)
                _, test_prediction = torch.max(outputs, 1)
                prediction = np.concatenate((prediction, test_prediction.cpu().numpy()), axis=0)
            print(prediction)
            print(prediction.shape)
