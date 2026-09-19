"""Differentiable trilinear reconstruction in a prediction-defined 3D grid.

Point tensors are [B,N,3,P] in image-0 millimetres. Volumes use [B,C,Z,Y,X].
Bounds never depend on tracker labels, including at training time.
"""
import itertools
import torch
from torch.nn import functional as F


def volume_bounds(points, margin_mm=10.):
    flat = points.detach().permute(0, 1, 3, 2).reshape(points.shape[0], -1, 3)
    low, high = flat.amin(1), flat.amax(1)
    return low - margin_mm, high + margin_mm


def normalize_points(points, bounds):
    low, high = bounds
    xyz = points.permute(0, 1, 3, 2)
    return 2 * (xyz - low[:, None, None]) / (high - low)[:, None, None] - 1


def splat(points, frames, bounds, shape, return_sums=False):
    """Eight-neighbour weighted intensity accumulation; gradients reach points."""
    b = points.shape[0]
    d, h, w = shape
    xyz = (normalize_points(points, bounds).reshape(b, -1, 3) + 1) / 2
    xyz = xyz * xyz.new_tensor([w - 1, h - 1, d - 1])
    lower = xyz.floor().long()
    fraction = xyz - lower
    values = frames.reshape(b, -1)
    intensity = values.new_zeros(b, d * h * w)
    weight_sum = torch.zeros_like(intensity)
    for dx, dy, dz in itertools.product((0, 1), repeat=3):
        delta = lower.new_tensor([dx, dy, dz])
        idx = lower + delta
        valid = ((idx >= 0) & (idx < idx.new_tensor([w, h, d]))).all(-1)
        weight = torch.where(delta.bool(), fraction, 1 - fraction).prod(-1) * valid
        address = (idx[..., 2].clamp(0, d - 1) * h + idx[..., 1].clamp(0, h - 1)) * w + idx[..., 0].clamp(0, w - 1)
        intensity = intensity.scatter_add(1, address, weight * values)
        weight_sum = weight_sum.scatter_add(1, address, weight)
    intensity = intensity.reshape(b, 1, d, h, w)
    weight_sum = weight_sum.reshape_as(intensity)
    if return_sums:
        return intensity, weight_sum
    return intensity / weight_sum.clamp_min(1e-6), weight_sum


def sample_field(field, points, bounds):
    grid = normalize_points(points, bounds).unsqueeze(3)
    return F.grid_sample(field, grid, align_corners=True, padding_mode='border').squeeze(-1).permute(0, 2, 1, 3)


def anchor_points(points, base, field, bounds):
    # The first image is the fixed global reference, including its full plane.
    return points + sample_field(field, points, bounds) - sample_field(field, base, bounds)


def warp_volume(volume, field, bounds):
    """Pull GT at x + field(x); field channels are x,y,z displacements in mm."""
    b, _, d, h, w = volume.shape
    z, y, x = torch.meshgrid(*(torch.linspace(-1, 1, n, device=volume.device,
                                             dtype=volume.dtype) for n in (d, h, w)), indexing='ij')
    grid = torch.stack((x, y, z), -1).unsqueeze(0).expand(b, -1, -1, -1, -1)
    low, high = bounds
    grid = grid + 2 * field.permute(0, 2, 3, 4, 1) / (high - low)[:, None, None, None]
    return F.grid_sample(volume, grid, align_corners=True, padding_mode='zeros')


def bending_energy(field, bounds):
    """Pure and mixed second spatial derivatives, with physical grid spacing."""
    low, high = bounds
    spacing = (high - low) / (field.new_tensor(list(reversed(field.shape[-3:]))) - 1)
    energy = field.new_zeros(())
    for axis, dim in enumerate((4, 3, 2)):
        second = field.diff(n=2, dim=dim) / spacing[:, axis, None, None, None, None].square()
        energy = energy + second.square().mean()
        for other in range(axis):
            mixed = field.diff(dim=dim).diff(dim=(4, 3, 2)[other])
            mixed = mixed / (spacing[:, axis] * spacing[:, other])[:, None, None, None, None]
            energy = energy + 2 * mixed.square().mean()
    return energy
