import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model


class ensembleNet(nn.Module):
    def __init__(self, model_names, resnet):
        super().__init__()
        self.models = nn.ModuleList([ptcv_get_model(name, pretrained=True) for name in model_names])
        self.models.append(resnet)

    def forward(self, x):
        legit = None
        for i, m in enumerate(self.models):
            legit = m(x) if i == 0 else legit + m(x)
        return legit / len(self.models)
