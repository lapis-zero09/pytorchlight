import torch.nn as nn


__all__ = ['CNN', 'cnn']


class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.fc = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


''' future function
def make_layers_for_vgg(num_conv=3, in_channels=3, batch_norm=False):
    layers = []
    for v in range(num_conv):
        conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
        if batch_norm:
            layers.append([conv2d,
                           nn.BatchNorm2d(v),
                           nn.ReLU(inplace=True)])
        else:
            layers.append([conv2d,
                           nn.ReLU(inplace=True),
                           nn.MaxPool2d(kernel_size=2)])
        in_channels = v

    return nn.Sequential(*layers)
'''


def cnn(in_channels=1, num_classes=10):
    model = CNN(in_channels, num_classes)
    return model
