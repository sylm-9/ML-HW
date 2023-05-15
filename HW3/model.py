import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
        )

        self.relu = nn.ReLU(inplace=True)
        self.down_sample = None
        if stride != 1 or (in_channels != out_channels):
            self.down_sample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.down_sample:
            residual = self.down_sample(x)
        out += residual
        return self.relu(out)


def make_residual(in_channels, out_channels, num_layer, stride=1):
    layers = [ResidualBlock(in_channels, out_channels, stride)]
    for i in range(1, num_layer):
        layers.append(ResidualBlock(out_channels, out_channels))
    return nn.Sequential(*layers)


class Classifier(nn.Module):
    def __init__(self, num_layers=None):
        super().__init__()
        if num_layers is None:
            num_layers = [2, 3, 3, 1]
        self.pre_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.layer0 = make_residual(32, 64, num_layers[0], stride=2)
        self.layer1 = make_residual(64, 128, num_layers[1], stride=2)
        self.layer2 = make_residual(128, 256, num_layers[2], stride=2)
        self.layer3 = make_residual(256, 512, num_layers[3], stride=2)

        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 11),
        )

    def forward(self, x):
        out = self.pre_conv(x)
        out = self.layer0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.fc(out.view(out.size(0), -1))
        return out
