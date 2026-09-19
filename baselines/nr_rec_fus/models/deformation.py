"""Compact 3D registration U-Net; only the predicted volume is an input."""
import torch
from torch import nn
from torch.nn import functional as F


def block(inputs, outputs):
    return nn.Sequential(nn.Conv3d(inputs, outputs, 3, padding=1), nn.LeakyReLU(.2),
                         nn.Conv3d(outputs, outputs, 3, padding=1), nn.LeakyReLU(.2))


class DeformationNet(nn.Module):
    def __init__(self, channels=8, max_displacement_mm=5.):
        super().__init__()
        self.enc = block(1, channels)
        self.down = block(channels, channels * 2)
        self.bottom = block(channels * 2, channels * 4)
        self.up1 = block(channels * 6, channels * 2)
        self.up0 = block(channels * 3, channels)
        self.head = nn.Conv3d(channels, 3, 3, padding=1)
        nn.init.normal_(self.head.weight, std=1e-5)
        nn.init.zeros_(self.head.bias)
        self.max_displacement_mm = max_displacement_mm

    def forward(self, volume):
        e0 = self.enc(volume)
        e1 = self.down(F.avg_pool3d(e0, 2))
        x = self.bottom(F.avg_pool3d(e1, 2))
        x = self.up1(torch.cat((F.interpolate(x, size=e1.shape[-3:], mode='trilinear', align_corners=True), e1), 1))
        x = self.up0(torch.cat((F.interpolate(x, size=e0.shape[-3:], mode='trilinear', align_corners=True), e0), 1))
        return self.max_displacement_mm * self.head(x).tanh()
