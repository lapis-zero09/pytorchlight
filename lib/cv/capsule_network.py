import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


__all__ = ['squash', 'CapsuleNetwork', 'CapsuleLoss', 'capsule_network']


def squash(input_tensor):
    squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
    output_tensor = squared_norm * input_tensor \
        / ((1. + squared_norm) * torch.sqrt(squared_norm))
    return output_tensor


class PrimaryCaps(nn.Module):
    def __init__(self, num_caps, in_channels=256, out_channels=32):
        super(PrimaryCaps, self).__init__()
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=9,
                      stride=2, padding=0) for _ in range(num_caps)
        ])

    def forward(self, x):
        u = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
        u = torch.cat(u, dim=-1)
        return squash(u)


class DigitCaps(nn.Module):
    def __init__(self, use_gpu, num_routing_iters,
                 in_channels=8, out_channels=16,
                 num_route_nodes=32 * 6 * 6, num_classes=10):
        super(DigitCaps, self).__init__()
        self.use_gpu = use_gpu
        self.num_iters = num_routing_iters
        self.weights = nn.Parameter(
            torch.randn(num_classes, num_route_nodes,
                        in_channels, out_channels)
        )

    def forward(self, x):
        u_ = torch.matmul(
            x[None, :, :, None, :], self.weights[:, None, :, :, :])
        if self.use_gpu:
            b = Variable(torch.zeros(*u_.shape)).cuda()
        else:
            b = Variable(torch.zeros(*u_.shape))

        # routing alg.
        for i in range(self.num_iters):
            c = F.softmax(b, dim=2)
            s = torch.mul(u_, c).sum(dim=2, keepdim=True)
            v = squash(s)

            if i < self.num_iters - 1:
                b = b + torch.mul(u_, c).sum(dim=-1, keepdim=True)

        return v.squeeze().transpose(0, 1)


class CapsuleNetwork(nn.Module):
    def __init__(self, in_channels, num_classes,
                 num_caps, num_routing_iters, use_gpu):
        super(CapsuleNetwork, self).__init__()
        self.use_gpu = use_gpu
        self.conv = nn.Conv2d(in_channels, 256, kernel_size=9)
        self.primary_caps = PrimaryCaps(num_caps)
        self.disit_caps = DigitCaps(use_gpu, num_routing_iters)

        self.decoder = nn.Sequential(nn.Linear(16 * num_classes, 512),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, 1024),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(1024, 28 * 28),
                                     nn.Sigmoid())

    def forward(self, x, y=None):
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.primary_caps(x)
        x = self.disit_caps(x)

        cls = torch.sqrt((x**2).sum(dim=-1))
        cls = F.softmax(cls, dim=-1)

        if y is None:
            _, idx = cls.max(dim=1)
            if self.use_gpu:
                y = Variable(
                    torch.eye(10)).cuda().index_select(dim=0, index=idx)
            else:
                y = Variable(torch.eye(10)).index_select(dim=0, index=idx)

        reconstr = self.decoder(
            torch.mul(x, y[:, :, None]).view(x.size(0), -1))

        return cls, reconstr


class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()
        self.mse_loss = nn.MSELoss(size_average=False)

    def margin_loss(self, pred, y_true):
        left = F.relu(0.9 - pred, inplace=True) ** 2
        right = F.relu(pred - 0.1, inplace=True) ** 2
        margin_loss = y_true * left + 0.5 * (1. - y_true) * right
        return margin_loss.sum()

    def reconstr_loss(self, reconstr, org_data):
        return self.mse_loss(reconstr, org_data) * 0.0005

    def forward(self, org_data, y_true, pred, reconstr):
        org_data = org_data.view(reconstr.size(0), -1)
        margin_loss = self.margin_loss(pred, y_true)
        reconstr_loss = self.reconstr_loss(reconstr, org_data)
        return (margin_loss + reconstr_loss) / org_data.size(0)


def capsule_network(in_channels=1, num_classes=10,
                    num_caps=8, num_routing_iters=3, use_gpu=False):
    if use_gpu:
        if not torch.cuda.is_available():
            use_gpu = False
            print("oops! Something is wrong.\n can't use gpu!\n Then use cpu.",
                  sep=' ', end='n', file=sys.stdout, flush=False)

    model = CapsuleNetwork(in_channels, num_classes,
                           num_caps, num_routing_iters, use_gpu)
    return model
