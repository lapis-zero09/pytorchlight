import torch.cat as torch_cat
import torch.Tensor as torch_tensor
import torch.nn as nn
from torch.nn.functional import F


class Inception(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Inception, self).__init__()
        self.base = BASE(in_channels)
        self.mixed0 = NIN_A(in_channels=192, out_channels_of_pool=32)
        self.mixed1 = NIN_A(in_channels=256, out_channels_of_pool=64)
        self.mixed2 = NIN_A(in_channels=288, out_channels_of_pool=64)

        self.mixed3 = NIN_B(in_channels=288)

        self.mixed4 = NIN_C(768, in_channels_7x7=128)
        self.mixed5 = NIN_C(768, in_channels_7x7=160)
        self.mixed6 = NIN_C(768, in_channels_7x7=160)
        self.mixed7 = NIN_C(768, in_channels_7x7=192)

        self.mixed8 = NIN_D(768)

        self.mixed9 = NIN_E(1280)
        self.mixed10 = NIN_E(2048)
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch_tensor(X.rvs(m.weight.data.numel()))
                values = values.view(m.weight.data.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.base(x)  # 35 x 35 x 192

        x = self.mixed0(x)  # 35 x 35 x 256
        x = self.mixed1(x)  # 35 x 35 x 288
        x = self.mixed2(x)  # 35 x 35 x 288

        x = self.mixed3(x)  # 17 x 17 x 768

        x = self.mixed4(x)  # 17 x 17 x 768
        x = self.mixed5(x)  # 17 x 17 x 768
        x = self.mixed6(x)  # 17 x 17 x 768
        x = self.mixed7(x)  # 17 x 17 x 768

        x = self.mixed8(x)  # 8 x 8 x 1280

        x = self.mixed9(x)  # 8 x 8 x 2048
        x = self.mixed10(x)  # 8 x 8 x 2048

        x = F.avg_pool2d(x, kernel_size=8)
        x = F.dropout(x, training=self.training)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class BNConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kargs):
        super(BNConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        return x


class BASE(nn.Module):
    def __init__(self, in_channels):
        super(BASE, self).__init__()
        self.conv2d_bn_1 = BNConv2d(in_channels, 32, kernel_size=3, stride=2)
        self.conv2d_bn_2 = BNConv2d(32, 32, kernel_size=3)
        self.conv2d_bn_3 = BNConv2d(32, 64, kernel_size=3, padding=1)
        self.conv2d_bn_4 = BNConv2d(64, 80, kernel_size=1)
        self.conv2d_bn_5 = BNConv2d(80, 192, kernel_size=3)

    def forward(self, x):
        x = self.conv2d_bn_1(x)
        x = self.conv2d_bn_2(x)
        x = self.conv2d_bn_3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)

        x = self.conv2d_bn_4(x)
        x = self.conv2d_bn_5(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        return x


class NIN_A(nn.Module):
    def __init__(self, in_channels, out_channels_of_pool):
        super(NIN_A, self).__init__()
        self.branch1x1 = BNConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5 = nn.Sequential(
            BNConv2d(in_channels, 48, kernel_size=1),
            BNConv2d(48, 64, kernel_size=5, padding=2)
        )

        self.branch3x3dbl = nn.Sequential(
            BNConv2d(in_channels, 64, kernel_size=1),
            BNConv2d(64, 96, kernel_size=3, padding=1),
            BNConv2d(96, 96, kernel_size=3, padding=1)
        )

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BNConv2d(in_channels, out_channels_of_pool, kernel_size=1)
        )

    def forward(self, x):
        out = [self.branch1x1(x),
               self.branch5x5(x),
               self.branch3x3dbl(x),
               self.branch_pool(x)]
        return torch_cat(out, 1)


class NIN_B(nn.Module):
    def __init__(self, in_channels):
        super(NIN_B, self).__init__()
        self.branch3x3 = BNConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl = nn.Sequential(
            BNConv2d(in_channels, 64, kernel_size=1),
            BNConv2d(64, 96, kernel_size=3, padding=1),
            BNConv2d(96, 96, kernel_size=3, stride=2)
        )

    def forward(self, x):
        out = [self.branch3x3(x),
               self.branch3x3dbl(x),
               F.max_pool2d(x, kernel_size=3, stride=2)]
        return torch_cat(out, 1)


class NIN_C(nn.Module):
    def __init__(self, in_channels, in_channels_7x7):
        super(NIN_C, self).__init__()
        self.branch1x1 = BNConv2d(in_channels, 192, kernel_size=1)

        self.branch7x7 = nn.Sequential(
            BNConv2d(in_channels, in_channels_7x7, kernel_size=1),
            BNConv2d(in_channels_7x7, in_channels_7x7,
                     kernel_size=(1, 7), padding=(0, 3)),
            BNConv2d(in_channels_7x7, 192,
                     kernel_size=(7, 1), padding=(3, 0))
        )

        self.branch7x7dbl = nn.Sequential(
            BNConv2d(in_channels, in_channels_7x7, kernel_size=1),
            BNConv2d(in_channels_7x7, in_channels_7x7,
                     kernel_size=(7, 1), padding=(3, 0)),
            BNConv2d(in_channels_7x7, in_channels_7x7,
                     kernel_size=(1, 7), padding=(0, 3)),
            BNConv2d(in_channels_7x7, in_channels_7x7,
                     kernel_size=(7, 1), padding=(3, 0)),
            BNConv2d(in_channels_7x7, 192,
                     kernel_size=(1, 7), padding=(0, 3))
        )

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BNConv2d(in_channels, 192, kernel_size=1)
        )

    def forward(self, x):
        out = [self.branch1x1(x),
               self.branch7x7(x),
               self.branch7x7dbl(x),
               self.branch_pool(x)]
        return torch_cat(out, 1)


class NIN_D(nn.Module):
    def __init__(self, in_channels):
        super(NIN_D, self).__init__()
        self.branch3x3 = nn.Sequential(
            BNConv2d(in_channels, 192, kernel_size=1),
            BNConv2d(192, 320, kernel_size=3, stride=2)
        )

        self.branch7x7x3 = nn.Sequential(
            BNConv2d(in_channels, 192, kernel_size=1),
            BNConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3)),
            BNConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0)),
            BNConv2d(192, 192, kernel_size=3, stride=2)
        )

    def forward(self, x):
        out = [self.branch3x3(x),
               self.branch7x7x3(x),
               F.max_pool2d(x, kernel_size=3, stride=2)]
        return torch_cat(out, 1)


class NIN_E(nn.Module):
    def __init__(self, in_channels):
        super(NIN_E, self).__init__()
        self.branch1x1 = BNConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BNConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BNConv2d(384, 384,
                                     kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BNConv2d(384, 384,
                                     kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BNConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BNConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BNConv2d(384, 384,
                                        kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BNConv2d(384, 384,
                                        kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BNConv2d(in_channels, 192, kernel_size=1)
        )

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [self.branch3x3_2a(branch3x3),
                     self.branch3x3_2b(branch3x3)]
        branch3x3 = torch_cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [self.branch3x3dbl_3a(branch3x3dbl),
                        self.branch3x3dbl_3b(branch3x3dbl)]
        branch3x3dbl = torch_cat(branch3x3dbl, 1)

        out = [self.branch1x1(x),
               branch3x3,
               branch3x3dbl,
               self.branch_pool(x)]
        return torch_cat(out, 1)


def inception_v3(in_channels=3, num_classes=1000):
    model = Inception(in_channels, num_classes)
    return model
