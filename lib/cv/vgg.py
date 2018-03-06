import torch.nn as nn
from math import sqrt


__all__ = ['VGG', 'vgg11', 'vgg13', 'vgg16', 'vgg19']


class VGG(nn.Module):
    def __init__(self, convs, num_classes):
        super(VGG, self).__init__()
        self.convs = convs

        self.classifier = nn.Sequential(
            nn.Linear(in_features=512 * 7 * 7,
                      out_features=4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def make_layers_for_vgg(cfg, in_channels, batch_norm):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)


def vgg11(in_channels=3, num_classes=1000, batch_norm=False):
    convs = make_layers_for_vgg(cfg['A'], in_channels, batch_norm)
    model = VGG(convs, num_classes)
    return model


def vgg13(in_channels=3, num_classes=1000, batch_norm=False):
    convs = make_layers_for_vgg(cfg['B'], in_channels, batch_norm)
    model = VGG(convs, num_classes)
    return model


def vgg16(in_channels=3, num_classes=1000, batch_norm=False):
    convs = make_layers_for_vgg(cfg['D'], in_channels, batch_norm)
    model = VGG(convs, num_classes)
    return model


def vgg19(in_channels=3, num_classes=1000, batch_norm=False):
    convs = make_layers_for_vgg(cfg['E'], in_channels, batch_norm)
    model = VGG(convs, num_classes)
    return model
