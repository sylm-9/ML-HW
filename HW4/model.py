import torch.nn as nn
from conformer import ConformerBlock


class Classifier(nn.Module):
    def __init__(self, d_model=224, speaker_num=600, dropout=0.25):
        super().__init__()
        self.pre_net = nn.Linear(40, d_model)
        self.encoder = ConformerBlock(
            dim=d_model,
            dim_head=4,
            heads=4,
            ff_mult=4,
            conv_expansion_factor=2,
            conv_kernel_size=20,
            attn_dropout=dropout,
            ff_dropout=dropout,
            conv_dropout=dropout,
        )

        self.out_layer = nn.Sequential(
            nn.BatchNorm1d(d_model),
            nn.Linear(d_model, speaker_num),
        )

    def forward(self, mels):
        out = self.pre_net(mels)
        out = out.permute(1, 0, 2)
        out = self.encoder(out)
        out = out.transpose(0, 1)
        stats = out.mean(dim=1)

        out = self.out_layer(stats)
        return out