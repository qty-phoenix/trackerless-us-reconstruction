"""Compact MoGLo-Net implementation following the public US3D source.

The public model uses shared encoders, patch correlation, global-local attention,
two recurrent heads and two motion predictions. This implementation keeps those
operations while allowing the TUS 120x160 input and configurable sequence length.
"""
import torch
from torch import nn
from torch.nn import functional as F
from baselines.nr_rec_fus.models.rigid_pose import se3_from_vec


class ResidualBlock(nn.Module):
    def __init__(self, channels, stride=1):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels, 3, stride, 1, bias=False),
                                  nn.BatchNorm2d(channels), nn.SiLU(),
                                  nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(channels))
        self.skip = nn.Conv2d(channels, channels, 1, stride) if stride != 1 else nn.Identity()

    def forward(self, x):
        return F.silu(self.body(x) + self.skip(x))


class SharedEncoder(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, base, 7, 2, 3, bias=False),
                                 nn.BatchNorm2d(base), nn.SiLU(),
                                 ResidualBlock(base), ResidualBlock(base, 2),
                                 nn.Conv2d(base, base * 2, 3, 2, 1, bias=False),
                                 nn.BatchNorm2d(base * 2), nn.SiLU(),
                                 ResidualBlock(base * 2), ResidualBlock(base * 2))

    def forward(self, x):
        return self.net(x)


class GlobalLocalAttention(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.local = nn.Sequential(nn.Linear(channels, channels), nn.SiLU(), nn.Linear(channels, channels))
        self.gate = nn.Linear(channels * 2, channels)

    def forward(self, local, global_token):
        sequence = self.norm(local + global_token)
        attended, weights = self.attn(sequence, sequence, sequence, need_weights=True)
        local = local + attended + self.local(local)
        gate = torch.sigmoid(self.gate(torch.cat((local, global_token), -1)))
        return gate * local + (1 - gate) * global_token, weights


class PatchCorrelation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(channels * 4 + 1, channels * 2, 3, padding=1),
                                  nn.BatchNorm2d(channels * 2), nn.SiLU(),
                                  nn.Conv2d(channels * 2, channels * 2, 3, padding=1), nn.SiLU())

    def forward(self, first, second):
        f1 = F.normalize(first, dim=1)
        f2 = F.normalize(second, dim=1)
        corr = (f1 * f2).sum(1, keepdim=True)
        return self.proj(torch.cat((first, second, (first - second).abs(), first * second, corr), 1))


def matrix_to_axis_angle(matrix):
    """Differentiable rotation matrix to axis-angle vector, stable near identity."""
    trace = matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
    cosine = ((trace - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
    angle = torch.acos(cosine)
    skew = torch.stack((matrix[..., 2, 1] - matrix[..., 1, 2],
                        matrix[..., 0, 2] - matrix[..., 2, 0],
                        matrix[..., 1, 0] - matrix[..., 0, 1]), -1)
    scale = angle / (2 * torch.sin(angle).clamp_min(1e-6))
    return skew * scale.unsqueeze(-1)


def transforms_to_motion(transforms):
    return torch.cat((transforms[..., :3, 3], matrix_to_axis_angle(transforms[..., :3, :3])), -1)


class MoGLoNet(nn.Module):
    def __init__(self, num_samples=5, base=32):
        super().__init__()
        if num_samples < 3:
            raise ValueError('MoGLo-Net requires at least three frames')
        self.num_samples = num_samples
        self.pairs = num_samples - 1
        channels = base * 2
        self.encoder = SharedEncoder(base)
        self.correlation = PatchCorrelation(channels)
        self.local_proj = nn.Linear(channels * 2, channels * 4)
        self.global_proj = nn.Linear(channels * 2, channels * 4)
        self.attention = GlobalLocalAttention(channels * 4)
        self.lstm_local = nn.LSTM(channels * 4, channels * 4, batch_first=True)
        self.lstm_global = nn.LSTM(channels * 4, channels * 4, batch_first=True)
        self.head_local = nn.Linear(channels * 4, 6)
        self.head_global = nn.Linear(channels * 4, 6)

    def forward(self, frames):
        if frames.shape[1] != self.num_samples:
            raise ValueError(f'Expected {self.num_samples} frames, got {frames.shape[1]}')
        b, s, c, h, w = frames.shape
        encoded = self.encoder(frames.reshape(b * s, c, h, w))
        features = encoded.reshape(b, s, encoded.shape[1], encoded.shape[2], encoded.shape[3])
        pair_features = []
        for i in range(self.pairs):
            pair_features.append(self.correlation(features[:, i], features[:, i + 1]))
        pair_features = torch.stack(pair_features, 1)
        local = pair_features.mean((-2, -1))
        global_token = pair_features.amax((-2, -1))
        local = self.local_proj(local)
        global_token = self.global_proj(global_token)
        fused, attention = self.attention(local, global_token)
        local_seq, _ = self.lstm_local(fused)
        global_seq, _ = self.lstm_global(fused + global_token)
        return torch.stack((self.head_local(local_seq), self.head_global(global_seq)), 1), fused, attention

    def transforms(self, frames, motion_scale):
        predictions, embedding, attention = self(frames)
        return se3_from_vec(predictions * motion_scale), embedding, attention
