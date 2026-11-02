"""3D ResNeXt-50 with spatial-temporal attention for DC2-Net.

The paper specifies a 3D ResNeXt-50 backbone and an attention map generated
from its final residual feature map.  The authors did not release source code,
so the attention branch below uses the conventional residual sigmoid gating
shown in the paper: a 1x1x1 bottleneck predicts one attention weight per
spatial-temporal location before global pooling and six-DoF regression.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ResNeXtBottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, groups=32, base_width=4):
        super().__init__()
        width = int(channels * (base_width / 64.0)) * groups
        spatial_stride = (1, stride, stride)
        self.body = nn.Sequential(
            nn.Conv3d(in_channels, width, 1, bias=False),
            nn.BatchNorm3d(width), nn.ReLU(inplace=True),
            nn.Conv3d(width, width, 3, spatial_stride, 1, groups=groups, bias=False),
            nn.BatchNorm3d(width), nn.ReLU(inplace=True),
            nn.Conv3d(width, channels * self.expansion, 1, bias=False),
            nn.BatchNorm3d(channels * self.expansion))
        self.skip = (nn.Identity() if stride == 1 and in_channels == channels * self.expansion
                     else nn.Sequential(
                         nn.Conv3d(in_channels, channels * self.expansion, 1,
                                   spatial_stride, bias=False),
                         nn.BatchNorm3d(channels * self.expansion)))

    def forward(self, value):
        return F.relu(self.body(value) + self.skip(value), inplace=True)


class DC2Net(nn.Module):
    """Return one mean adjacent motion vector for each input video segment."""

    def __init__(self, layers=(3, 4, 6, 3), groups=32, base_width=4):
        super().__init__()
        if len(layers) != 4 or min(layers) < 1:
            raise ValueError('DC2-Net requires four nonempty residual stages')
        self.groups, self.base_width = groups, base_width
        self.in_channels = 64
        self.stem = nn.Sequential(
            nn.Conv3d(1, 64, (3, 7, 7), (1, 2, 2), (1, 3, 3), bias=False),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 3, 3), (1, 2, 2), (0, 1, 1)))
        self.layer1 = self._stage(64, layers[0], 1)
        self.layer2 = self._stage(128, layers[1], 2)
        self.layer3 = self._stage(256, layers[2], 2)
        self.layer4 = self._stage(512, layers[3], 2)
        # A compact attention branch preserves the temporal axis and predicts a
        # single saliency map over speckle-rich locations.
        self.attention = nn.Sequential(
            nn.Conv3d(2048, 256, 1, bias=False), nn.BatchNorm3d(256),
            nn.ReLU(inplace=True), nn.Conv3d(256, 1, 1), nn.Sigmoid())
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(2048, 6)
        self._initialize()

    def _stage(self, channels, blocks, stride):
        result = [ResNeXtBottleneck3D(self.in_channels, channels, stride,
                                      self.groups, self.base_width)]
        self.in_channels = channels * ResNeXtBottleneck3D.expansion
        result.extend(ResNeXtBottleneck3D(self.in_channels, channels, 1,
                                          self.groups, self.base_width)
                      for _ in range(1, blocks))
        return nn.Sequential(*result)

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.head.bias)

    def forward(self, frames):
        if frames.ndim != 4 or frames.shape[1] < 2:
            raise ValueError('DC2-Net input must be [B,T,H,W] with T >= 2')
        # Per-video intensity normalization follows the paper's normalization
        # objective while avoiding any validation-set statistics.
        value = frames[:, None]
        value = (value - value.mean((-3, -2, -1), keepdim=True)) / (
            value.std((-3, -2, -1), keepdim=True) + 1e-6)
        value = self.layer4(self.layer3(self.layer2(self.layer1(self.stem(value)))))
        attention = self.attention(value)
        embedding = self.pool(value * (1 + attention)).flatten(1)
        return self.head(embedding), embedding, attention
