import os
import json
from dataset import TestDataset
import torch
from torch.utils.data import DataLoader
from model import Classifier


def collate_batch(batch):
    batch_feat_names, batch_mels = zip(*batch)
    return batch_feat_names, torch.stack(batch_mels)


if __name__ == "__main__":
    data_dir = "./Dataset"
    model_path = "model.ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mapping = json.load(open(os.path.join(data_dir, "mapping.json")))
    test_set = TestDataset(data_dir)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, drop_last=False, num_workers=4,
                             collate_fn=collate_batch)
    speaker_num = len(mapping["id2speaker"])
    model = Classifier(speaker_num=speaker_num).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    results = []
    for feat_names, mels in test_loader:
        with torch.no_grad():
            mels = mels.to(device)
            outputs = model(mels)
            predictions = outputs.argmax(1).cpu().numpy()
            for feat_name, prediction in zip(feat_names, predictions):
                results.append([feat_name, mapping["id2speaker"][str(prediction)]])
