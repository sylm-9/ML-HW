import torch.nn as nn


def dconv_bn_relu(in_dim, out_dim):
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
        nn.BatchNorm2d(out_dim),
        nn.ReLU(True)
    )


class Generator(nn.Module):
    def __init__(self, in_dim, feature_dim=64):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, feature_dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(feature_dim * 8 * 4 * 4),
            nn.ReLU()
        )
        self.l2 = nn.Sequential(
            dconv_bn_relu(feature_dim * 8, feature_dim * 4),
            dconv_bn_relu(feature_dim * 4, feature_dim * 2),
            dconv_bn_relu(feature_dim * 2, feature_dim),
        )
        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 3, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2(y)
        y = self.l3(y)
        return y


def conv_bn_lrelu(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 4, 2, 1),
        nn.InstanceNorm2d(out_dim),
        nn.LeakyReLU(0.2),
    )


class Discriminator(nn.Module):

    def __init__(self, in_dim, feature_dim=64):
        super(Discriminator, self).__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(in_dim, feature_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            conv_bn_lrelu(feature_dim, feature_dim * 2),
            conv_bn_lrelu(feature_dim * 2, feature_dim * 4),
            conv_bn_lrelu(feature_dim * 4, feature_dim * 8),
            nn.Conv2d(feature_dim * 8, 1, kernel_size=4, stride=1, padding=0),
        )
        self.apply(weights_init)

    def forward(self, x):
        y = self.l1(x)
        y = y.view(-1)
        return y


def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        model.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)