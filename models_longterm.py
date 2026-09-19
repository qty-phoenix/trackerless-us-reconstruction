"""EfficientNet-B1 and geometric point supervision from the published baseline."""
import math
import torch
from torch import nn
from torchvision.models import efficientnet_b1


def pair_samples(num_samples=10, num_pred=9):
    return torch.tensor([[n0, n1] for n1 in range(num_samples - num_pred, num_samples) for n0 in range(n1)], dtype=torch.long)


def reference_image_points(height=120, width=160):
    # x/y pixel coordinates, four corners, homogeneous row included.
    x, y = torch.meshgrid(torch.tensor([0., width - 1.]), torch.tensor([0., height - 1.]), indexing='xy')
    points = torch.stack([x.reshape(-1), y.reshape(-1), torch.zeros(4), torch.ones(4)])
    return points


def relative_tool_transforms(tforms):
    """T(tool_j -> tool_i) for every configured pair; shape [B,45,4,4]."""
    pairs = pair_samples().to(tforms.device)
    inv = torch.linalg.inv(tforms)
    return inv[:, pairs[:, 0]] @ tforms[:, pairs[:, 1]]


def transform_points(transforms, image_to_tool, points):
    # transforms [B,P,4,4], points [4,K] -> [B,P,3,K]
    pts_tool = image_to_tool @ points.to(transforms.device)
    return (transforms @ pts_tool[None, None]).contiguous()[:, :, :3]


def euler_xyz_to_matrix(params):
    """Match freehand/transform.py: params (..., 6) = rz, ry, rx, tx, ty, tz."""
    sz, cz = torch.sin(params[..., 0]), torch.cos(params[..., 0])
    sy, cy = torch.sin(params[..., 1]), torch.cos(params[..., 1])
    sx, cx = torch.sin(params[..., 2]), torch.cos(params[..., 2])
    r = torch.stack([cy*cz, sx*sy*cz-cx*sz, cx*sy*cz+sx*sz,
                     cy*sz, sx*sy*sz+cx*cz, cx*sy*sz-sx*cz,
                     -sy, sx*cy, cx*cy], dim=-1).reshape(*params.shape[:-1], 3, 3)
    out = torch.zeros(*params.shape[:-1], 4, 4, device=params.device, dtype=params.dtype)
    out[..., :3, :3] = r
    out[..., :3, 3] = params[..., 3:]
    out[..., 3, 3] = 1
    return out


class LongTermEfficientNet(nn.Module):
    def __init__(self, num_pairs=45, pretrained=False):
        super().__init__()
        weights = None
        self.backbone = efficientnet_b1(weights=weights)
        first = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(10, first.out_channels, first.kernel_size,
                                                   first.stride, first.padding, bias=first.bias is not None)
        self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, num_pairs * 6)

    def forward(self, frames):
        return self.backbone(frames).reshape(frames.shape[0], 45, 6)
