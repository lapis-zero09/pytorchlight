import torch.nn as nn
import torch.nn.functional as F


__all__ = ['ResNeXt', 'resnext26', 'resnext50', 'resnext101']


class IdentityLayers_C(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=2,
                 cardinality=32, stride=1):
        super(IdentityLayers_C, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1, groups=cardinality,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * expansion,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * expansion)
        )

        self.shortcut = None
        if stride != 1 or in_channels != out_channels * expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels * expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels * expansion)
            )

    def forward(self, x):
        out = self.conv(x)

        if self.shortcut is not None:
            out += self.shortcut(x)
        else:
            out += x

        return F.relu(out, inplace=True)


class ResNeXt(nn.Module):
    def __init__(self, in_channels, num_classes, cfg,
                 cardinality=32, base_width=4):
        super(ResNeXt, self).__init__()
        self.expansion = 2
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, self.in_channels, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True)
        )

        self.cardinality = cardinality
        self.base_width = base_width
        self.conv2 = self._make_indentity_block(cfg[0])
        self.conv3 = self._make_indentity_block(cfg[1], stride=2)
        self.conv4 = self._make_indentity_block(cfg[2], stride=2)
        self.conv5 = self._make_indentity_block(cfg[3], stride=2)

        self.fc = nn.Linear(self.cardinality * self.base_width,
                            num_classes)

    def _make_indentity_block(self, num_layers, stride=1):
        out_channels = self.cardinality * self.base_width
        layers = []
        strides_for_layer = [stride] + [1] * (num_layers - 1)
        for stride in strides_for_layer:
            layers.append(
                IdentityLayers_C(in_channels=self.in_channels,
                                 out_channels=out_channels,
                                 cardinality=self.cardinality,
                                 stride=stride,
                                 expansion=self.expansion)
            )
            self.in_channels = self.expansion * out_channels

        self.base_width *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = F.avg_pool2d(x, kernel_size=4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


cfg = {
    '26': [2, 2, 2, 2],
    '50': [3, 4, 6, 3],
    '101': [3, 4, 23, 3]
}


def resnext26(in_channels=3, num_classes=1000, cardinality=32, base_width=4):
    model = ResNeXt(in_channels, num_classes,
                    cfg['26'], cardinality, base_width)
    return model


def resnext50(in_channels=3, num_classes=1000, cardinality=32, base_width=4):
    model = ResNeXt(in_channels, num_classes,
                    cfg['50'], cardinality, base_width)
    return model


def resnext101(in_channels=3, num_classes=1000, cardinality=32, base_width=4):
    model = ResNeXt(in_channels, num_classes,
                    cfg['101'], cardinality, base_width)
    return model
