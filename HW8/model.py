from torch import nn


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


class ResNet(nn.Module):
    def __init__(self, num_layers=None):
        super().__init__()
        if num_layers is None:
            num_layers = [2, 1, 1, 1]
        self.preconv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer0 = make_residual(32, 64, num_layers[0], stride=2)
        self.layer1 = make_residual(64, 128, num_layers[1], stride=2)
        self.layer2 = make_residual(128, 128, num_layers[2], stride=2)
        self.layer3 = make_residual(128, 64, num_layers[3], stride=2)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64 * 4 * 4, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 64 * 4 * 4),
            nn.BatchNorm1d(64 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (64, 4, 4)),
            nn.ConvTranspose2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encoder(self, x):
        x = self.preconv(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc(x)
        return x

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
