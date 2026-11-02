"""The shallow two-frame pose-regression CNN from Prevost et al. (2018)."""
from torch import nn


class TwoFrameCNN(nn.Module):
    """Regress one adjacent 6-DoF image transform from two stacked frames.

    The layer widths, kernels, strides and dropout follow the public
    Prevost2018ConvNet reproduction in DualTrack.  The original architecture
    has a fixed fully connected input and therefore expects 128 x 128 frames.
    """

    def __init__(self, dropout=0.25, input_size=128, input_width=None):
        super().__init__()
        self.input_height = input_size
        self.input_width = input_size if input_width is None else input_width
        if (self.input_height, self.input_width) not in ((128, 128), (120, 160)):
            raise ValueError('TwoFrameCNN supports the paper 128x128 or FairBench 120x160 input')
        self.conv1 = nn.Conv2d(2, 64, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(64, 64, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(64 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 6)

    def forward(self, frames):
        if frames.ndim != 4 or frames.shape[1:] != (2, self.input_height, self.input_width):
            raise ValueError(f'TwoFrameCNN input must be [B,2,{self.input_height},{self.input_width}]')
        value = self.relu(self.conv1(frames))
        value = self.relu(self.conv2(value))
        value = self.pool(self.dropout(value))
        value = self.relu(self.conv3(value))
        value = self.relu(self.conv4(value))
        value = self.pool(self.dropout(value))
        value = value.flatten(1)
        value = self.dropout(self.relu(self.fc1(value)))
        return self.fc2(value)
