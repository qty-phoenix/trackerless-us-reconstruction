"""RecON networks and motion/image helpers.

The backbone follows the public implementation: six-channel adjacent-frame
inputs, a shared ResNet-18 encoder, a spatial ConvLSTM and a six-DoF head.  The
global-shape discriminator is the public 3D ResNet-10 design.
"""
import cv2
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


def correlation_loss(target, prediction, eps=1e-6):
    x, y = prediction.flatten(1), target.flatten(1)
    x, y = x - x.mean(1, keepdim=True), y - y.mean(1, keepdim=True)
    correlation = (x * y).mean(1) / (x.square().mean(1).add(eps).sqrt() *
                                     y.square().mean(1).add(eps).sqrt())
    return 1 - correlation.mean()


def scaled_motion(value, angle_scale):
    return torch.cat((value[..., :3], value[..., 3:] * angle_scale), -1)


def unscaled_motion(value, angle_scale):
    return torch.cat((value[..., :3], value[..., 3:] / angle_scale), -1)


def dof_to_matrix(dof):
    """Translation plus XYZ Euler angles, matching the authors' convention."""
    translation, angles = dof[..., :3], dof[..., 3:]
    ax, ay, az = angles.unbind(-1)
    sx, sy, sz = torch.sin(ax), torch.sin(ay), torch.sin(az)
    cx, cy, cz = torch.cos(ax), torch.cos(ay), torch.cos(az)
    rotation = torch.stack((cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz,
                            cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz,
                            -sy, sx * cy, cx * cy), -1).reshape(*dof.shape[:-1], 3, 3)
    result = torch.zeros(*dof.shape[:-1], 4, 4, device=dof.device, dtype=dof.dtype)
    result[..., :3, :3] = rotation
    result[..., :3, 3] = translation
    result[..., 3, 3] = 1
    return result


def matrix_to_dof(matrix, eps=1e-6):
    """Inverse of :func:`dof_to_matrix` away from Euler singularities."""
    rotation = matrix[..., :3, :3]
    cy = torch.sqrt(rotation[..., 0, 0].square() + rotation[..., 1, 0].square())
    ax = torch.atan2(rotation[..., 2, 1], rotation[..., 2, 2])
    ay = torch.atan2(-rotation[..., 2, 0], cy)
    az = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
    singular = cy <= eps
    ax = torch.where(singular, torch.atan2(-rotation[..., 1, 2], rotation[..., 1, 1]), ax)
    az = torch.where(singular, torch.zeros_like(az), az)
    return torch.cat((matrix[..., :3, 3], torch.stack((ax, ay, az), -1)), -1)


def chain_relative(relative):
    """Compose adjacent image transforms into first-frame-relative transforms."""
    identity = torch.eye(4, device=relative.device, dtype=relative.dtype)
    current = identity.expand(*relative.shape[:-3], 4, 4)
    values = [current]
    for index in range(relative.shape[-3]):
        current = current @ relative[..., index, :, :]
        values.append(current)
    return torch.stack(values, -3)


def _edge_images(frames):
    # Match the public Canny2D preprocessing: Gaussian blur, Sobel magnitude,
    # per-frame normalization and a 0.1 early threshold.
    coords = torch.arange(-2, 3, device=frames.device, dtype=frames.dtype)
    gaussian = torch.exp(-coords.square() / 2)
    gaussian = gaussian / gaussian.sum()
    horizontal = gaussian[None, None, None, :]
    vertical = gaussian[None, None, :, None]
    kernel_x = frames.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])[None, None]
    kernel_y = kernel_x.transpose(-1, -2)
    x = frames.reshape(-1, 1, *frames.shape[-2:])
    blurred = F.conv2d(F.conv2d(x, horizontal, padding=(0, 2)), vertical,
                       padding=(2, 0))
    magnitude = torch.sqrt(F.conv2d(blurred, kernel_x, padding=1).square() +
                           F.conv2d(blurred, kernel_y, padding=1).square() + 1e-12)
    low = magnitude.flatten(1).amin(1)[:, None, None, None]
    high = magnitude.flatten(1).amax(1)[:, None, None, None]
    magnitude = (magnitude - low) / (high - low).clamp_min(1e-6)
    magnitude = torch.where(magnitude >= .1, magnitude, torch.zeros_like(magnitude))
    return magnitude.reshape(*frames.shape[:-2], 1, *frames.shape[-2:])


def _farneback(frames):
    """Deterministic CPU fallback for the authors' CUDA OpenCV optical flow."""
    shape = frames.shape
    arrays = (frames.detach().cpu().clamp(0, 1).mul(255).to(torch.uint8).numpy()
              .reshape(-1, shape[-3], shape[-2], shape[-1]))
    batches = []
    for sequence in arrays:
        flows = []
        for first, second in zip(sequence[:-1], sequence[1:]):
            flow = cv2.calcOpticalFlowFarneback(first, second, None, .5, 3, 15, 3, 5, 1.2, 0)
            flows.append(torch.from_numpy(flow).permute(2, 0, 1))
        batches.append(torch.stack(flows))
    value = torch.stack(batches).reshape(*shape[:-3], shape[-3] - 1, 2, shape[-2], shape[-1])
    return value.to(device=frames.device, dtype=frames.dtype)


def pair_inputs(frames):
    """Return [B,T-1,6,H,W]: two frames, two edges and 2-D flow."""
    unbatched = frames.ndim == 3
    if unbatched:
        frames = frames.unsqueeze(0)
    if frames.ndim != 4 or frames.shape[1] < 2:
        raise ValueError('frames must be [B,T,H,W] with T >= 2')
    edges = _edge_images(frames).squeeze(-3)
    flow = _farneback(frames)
    result = torch.cat((frames[:, :-1, None], frames[:, 1:, None],
                        edges[:, :-1, None], edges[:, 1:, None], flow), 2)
    return result[0] if unbatched else result


class ConvLSTMCell(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gates = nn.Conv2d(channels * 2, channels * 4, 3, padding=1)

    def forward(self, value, state):
        hidden, cell = state
        ingate, forgetgate, candidate, outgate = self.gates(
            torch.cat((value, hidden), 1)).chunk(4, 1)
        cell = torch.sigmoid(forgetgate) * cell + torch.sigmoid(ingate) * torch.tanh(candidate)
        hidden = torch.sigmoid(outgate) * torch.tanh(cell)
        return hidden, cell


class RecONBackbone(nn.Module):
    def __init__(self, imagenet_pretrained=False):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if imagenet_pretrained else None
        encoder = resnet18(weights=weights)
        old = encoder.conv1
        encoder.conv1 = nn.Conv2d(6, old.out_channels, old.kernel_size, old.stride,
                                  old.padding, bias=False)
        if weights is not None:
            with torch.no_grad():
                # timm's in_chans adaptation repeats RGB weights and preserves
                # activation scale by multiplying by 3 / in_channels.
                encoder.conv1.weight[:, :3] = old.weight * .5
                encoder.conv1.weight[:, 3:] = old.weight * .5
        self.encoder = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool,
                                     encoder.layer1, encoder.layer2, encoder.layer3, encoder.layer4)
        self.recurrent = ConvLSTMCell(512)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(512, 6)

    def forward(self, inputs, return_feature=False):
        if inputs.ndim != 5 or inputs.shape[2] != 6:
            raise ValueError('RecON input must be [B,T,6,H,W]')
        b, t = inputs.shape[:2]
        normalized = ((inputs - inputs.mean((-2, -1), keepdim=True)) /
                      (inputs.std((-2, -1), keepdim=True) + 1e-6))
        encoded = self.encoder(normalized.reshape(b * t, 6, *inputs.shape[-2:]))
        encoded = encoded.reshape(b, t, *encoded.shape[1:])
        raw_feature = self.pool(encoded.reshape(b * t, *encoded.shape[2:])).reshape(b, t, -1)
        hidden = encoded.new_zeros(b, 512, *encoded.shape[-2:])
        cell = torch.zeros_like(hidden)
        outputs = []
        for index in range(t):
            hidden, cell = self.recurrent(encoded[:, index], (hidden, cell))
            outputs.append(self.head(self.pool(hidden).flatten(1)))
        prediction = torch.stack(outputs, 1)
        return prediction, raw_feature if return_feature else None


class Basic3DBlock(nn.Module):
    def __init__(self, input_channels, channels, stride=1):
        super().__init__()
        self.body = nn.Sequential(nn.Conv3d(input_channels, channels, 3, stride, 1, bias=False),
                                  nn.BatchNorm3d(channels), nn.ReLU(inplace=True),
                                  nn.Conv3d(channels, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm3d(channels))
        self.skip = (nn.Identity() if stride == 1 and input_channels == channels else
                     nn.Sequential(nn.Conv3d(input_channels, channels, 1, stride, bias=False),
                                   nn.BatchNorm3d(channels)))

    def forward(self, value):
        return F.relu(self.body(value) + self.skip(value), inplace=True)


class ShapeDiscriminator(nn.Module):
    """3D ResNet-10 discriminator used for global anatomical shape."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(1, 64, 7, (1, 2, 2), 3, bias=False),
                                  nn.BatchNorm3d(64), nn.ReLU(inplace=True),
                                  nn.MaxPool3d(3, 2, 1))
        self.layers = nn.Sequential(Basic3DBlock(64, 64), Basic3DBlock(64, 128, 2),
                                    Basic3DBlock(128, 256, 2), Basic3DBlock(256, 512, 2))
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(512, 1)

    def forward(self, volume):
        return self.head(self.pool(self.layers(self.stem(volume))).flatten(1))


class RecON(nn.Module):
    def __init__(self, imagenet_pretrained=False):
        super().__init__()
        self.backbone = RecONBackbone(imagenet_pretrained)
        self.discriminator = ShapeDiscriminator()

    def forward(self, inputs, return_feature=False):
        return self.backbone(inputs, return_feature)
