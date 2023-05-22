import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split, DataLoader

from dataset import MyDataset
from model import Classifier
from scheduler import get_cosine_schedule_with_warmup


def collate_batch(batch):
    mel, speaker = zip(*batch)
    mel = pad_sequence(mel, batch_first=True, padding_value=-20)
    return mel, torch.FloatTensor(speaker).long()


if __name__ == "__main__":
    data_dir = "./Dataset"
    model_path = "model.ckpt"
    batch_size = 32
    steps = 100000
    warmup_steps = 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MyDataset(data_dir)
    speaker_num = dataset.get_speaker_number()
    train_set, val_set = random_split(dataset, [int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True
                              , num_workers=4, pin_memory=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=4, pin_memory=True, collate_fn=collate_batch)

    model = Classifier(speaker_num=speaker_num)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=warmup_steps,num_training_steps=steps)
    step = 0
    best_acc = 0
    while step <= steps:
        train_acc = 0.0
        train_loss = 0.0
        for i, batch in enumerate(train_loader):
            model.train()
            mels, labels =batch
            optimizer.zero_grad()
            outputs = model(mels.to(device))
            labels=labels.to(device)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(1)
            acc = torch.mean((predictions == labels).float())
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_acc = acc.item()
            train_loss = loss.item()
            step += 1
            if step % 50 == 0:
                print('Step: {}, train_acc: {:.5f}, train_loss: {:.5f}'.format(step, train_acc, train_loss))
            if step % 2000 == 0:
                model.eval()
                val_acc = 0.0
                val_loss = 0.0
                for j, val_batch in enumerate(val_loader):
                    with torch.no_grad():
                        mels, labels = val_batch
                        optimizer.zero_grad()
                        outputs = model(mels.to(device))
                        labels = labels.to(device)
                        loss = criterion(outputs, labels)
                        predictions = outputs.argmax(1)
                        acc = torch.mean((predictions == labels).float())
                        val_loss += loss.item()
                        val_acc += acc.item()
                print('Step: {}, valid_acc: {:.5f}, valid_loss: {:.5f}'.format(step, val_acc/(j+1), val_loss/(j+1)))
                val_acc = val_acc/(j+1)
                if val_acc > best_acc:
                    best_acc = val_acc
                    torch.save(model.state_dict(), model_path)
                    print('saving model with acc {:.5f}'.format(best_acc))
