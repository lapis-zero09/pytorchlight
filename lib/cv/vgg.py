import touch.nn as nn
from math import sqrt


class VGG(nn.Module):
    def __init__(self, num_classes, convs):
        super(VGG, self).__init__()
        self.convs = convs
        self._initialize_weights()

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

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

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


def make_layers_for_vgg(cfg, in_channels, batch_norm=False):
    layers = []
    for v in cfg:
        if v == 'M':
            layers.append([nn.MaxPool2d(kernel_size=2, stride=2)])
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers.append([conv2d,
                               nn.BatchNorm2d(v),
                               nn.ReLU(inplace=True)])
            else:
                layers.append([conv2d, nn.ReLU(inplace=True)])
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
