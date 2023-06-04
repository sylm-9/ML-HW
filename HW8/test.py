import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler
from dataset import HW8Dataset
import pandas as pd
from PIL import Image

# batch_size = 200
test_data = np.load("./data/testingset.npy", allow_pickle=True)
# data = torch.tensor(test_data, dtype=torch.float32)
# test_dataset = HW8Dataset(data)
# test_sampler = SequentialSampler(test_dataset)
# test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size, num_workers=1)
# eval_loss = nn.MSELoss(reduction='none')
# checkpoint_path = 'best_model.pt'
# model = torch.load(checkpoint_path)
# model.eval()
# out_file = 'prediction.csv'
# anomaly = list()
# with torch.no_grad():
#     for i, data in enumerate(test_dataloader):
#         img = data.float().cuda()
#         output = model(img)
#         loss = eval_loss(output, img).sum([1, 2, 3])
#         anomaly.append(loss)
# anomaly = torch.cat(anomaly, axis=0)
# anomaly = torch.sqrt(anomaly).reshape(len(test_data), 1).cpu().numpy()
# df = pd.DataFrame(anomaly, columns=['score'])
# df.to_csv(out_file, index_label='ID')

ids = [6385, 1601, 3612, 686, 4986, 4230, 9361, 10628, 12023, 3185]
for i, id in enumerate(ids):
    image = test_data[id]
    save_path = f"{i}.jpg"
    img = Image.fromarray(image)
    img = img.resize((256, 256))
    img = img.convert('RGB')
    img.save(save_path)

