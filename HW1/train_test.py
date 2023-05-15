from torch.utils.data import DataLoader
from dataset import select_feat, MyDataset
from function import same_seed, train_valid_split
import torch
import pandas as pd
from model import MyModel
import math
import torch.nn as nn


if __name__ == '__main__':
    mode = 'test'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 256
    model_path = './model.ckpt'
    same_seed(1314)
    train_data, test_data = pd.read_csv('./train.csv').values, pd.read_csv('./test.csv').values
    train_data, valid_data = train_valid_split(train_data, 0.2, 1314)
    x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data)
    train_dataset, valid_dataset, test_dataset = MyDataset(x_train, y_train), MyDataset(x_valid, y_valid), MyDataset(
        x_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if mode == 'train':
        model = MyModel(input_dim=x_train.shape[1]).to(device)
        criterion = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        epochs = 3000
        best_loss = math.inf
        step = 0
        early_stop_count = 0

        for epoch in range(epochs):
            model.train()
            loss_record = []
            for x, y in train_loader:
                optimizer.zero_grad()
                x, y = x.to(device), y.to(device)
                prediction = model(x)
                loss = criterion(prediction, y)
                loss.backward()
                optimizer.step()
                step = step + 1
                loss_record.append(loss.detach().item())

            train_loss = sum(loss_record) / len(loss_record)
            model.eval()
            loss_record = []
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    prediction = model(x)
                    loss = criterion(prediction, y)
                loss_record.append(loss.item())

            valid_loss = sum(loss_record) / len(loss_record)
            print('Epoch: {}, Train_Loss: {:.5f}, Valid_Loss: {:.5f}'.format(epoch, train_loss, valid_loss))

            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(model.state_dict(), model_path)
                print('model updated with loss {:.5f}...'.format(best_loss))
                early_stop_count = 0
            else:
                early_stop_count += 1

            if early_stop_count >= 400:
                print('Model not updated in 400 epochs,stop')
                break
    elif mode == 'test':
        model = MyModel(input_dim=x_test.shape[1]).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        predictions = []
        for x in test_loader:
            x = x.to(device)
            with torch.no_grad():
                prediction = model(x)
                predictions.append(prediction.detach().cpu())
        predictions = torch.cat(predictions, dim=0).numpy()
        print(predictions)




