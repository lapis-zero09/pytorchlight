import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class IdentityLayers(nn.Module):
    def __init__(self, in_channels, out_channels, expansion,
                 stride=1, shortcut=None):
        super(IdentityLayers, self).__init__()
        self.stride = stride
        self.shortcut = shortcut

        if expansion == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3,
                          stride=self.stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3,
                          stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3,
                          stride=self.stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels * 4,
                          kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )

    def forward(self, x):
        out = self.conv(x)

        if self.shortcut is not None:
            out += self.shortcut(x)
        else:
            out += x

        return F.relu(out, inplace=True)


class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes, cfg):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.expansion = cfg[0]
        layers = cfg[1]

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.Relu(inplace=True)
        )

        self.conv2 = self._make_indentity_block(64, layers[0])
        self.conv3 = self._make_indentity_block(128, layers[1], stride=2)
        self.conv4 = self._make_indentity_block(256, layers[2], stride=2)
        self.conv5 = self._make_indentity_block(512, layers[3], stride=2)

        self.fc = nn.Linear(512 * self.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_indentity_block(self, out_channels, num_layers, stride=1):
        shortcut = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * self.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

        layers.append(
            IdentityLayers(self.in_channels, out_channels,
                           expansion=self.expansion,
                           stride=stride, shortcut=shortcut)
        )
        self.in_channels = out_channels * self.expansion

        for i in range(1, num_layers):
            layers.append(
                IdentityLayers(self.in_channels, out_channels,
                               expansion=self.expansion)
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = F.avg_pool2d(x, kernel_size=7, stride=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


cfg = {
    '18': [1, [2, 2, 2, 2]],
    '34': [1, [3, 4, 6, 3]],
    '50': [4, [3, 4, 6, 3]],
    '101': [4, [3, 4, 23, 3]],
    '152': [4, [3, 8, 36, 3]]
}


def resnet18(in_channels=3, num_classes=1000):
    model = ResNet(in_channels, num_classes, cfg['18'])
    return model


def resnet34(in_channels=3, num_classes=1000):
    model = ResNet(in_channels, num_classes, cfg['34'])
    return model


def resnet50(in_channels=3, num_classes=1000):
    model = ResNet(in_channels, num_classes, cfg['50'])
    return model


def resnet101(in_channels=3, num_classes=1000):
    model = ResNet(in_channels, num_classes, cfg['101'])
    return model


def resnet152(in_channels=3, num_classes=1000):
    model = ResNet(in_channels, num_classes, cfg['152'])
    return model
