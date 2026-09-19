"""TUS-REC2024 geometry, using the official submission's conventions.

Pixels: x=1..W, y=1..H, x varies fastest. Landmark frame k indexes scan[k]
(the official converter indexes the N-1 transform array at k-1).
Transforms here map tool_i to tool_0 unless explicitly named image transforms.
"""
from dataclasses import dataclass
from functools import cached_property
import numpy as np
import torch
from datasets.tus_rec2024 import load_calibration


def image_points(height=480, width=640, *, device=None, dtype=torch.float64):
    y, x = torch.meshgrid(torch.arange(1, height + 1, device=device, dtype=dtype),
                          torch.arange(1, width + 1, device=device, dtype=dtype), indexing='ij')
    return torch.stack((x.flatten(), y.flatten(), torch.zeros(height * width, device=device, dtype=dtype),
                        torch.ones(height * width, device=device, dtype=dtype)))


def calibration_tensors(path, device='cpu', dtype=torch.float32, size=(480, 640)):
    calib = load_calibration(path)
    scale = torch.tensor(calib['pixel_to_mm'], device=device, dtype=dtype)
    spatial = torch.tensor(calib['image_mm_to_tool'], device=device, dtype=dtype)
    resize = torch.diag(scale.new_tensor([640 / size[1], 480 / size[0], 1, 1]))
    return scale @ resize, spatial


def global_from_labels(tool_to_world):
    return torch.linalg.inv(tool_to_world[:1]) @ tool_to_world


def local_from_global(transforms):
    return torch.linalg.inv(transforms[:-1]) @ transforms[1:]


@dataclass
class ScanPrediction:
    """Compact reconstruction; optional NR field is in image-0 mm coordinates.

For NR, x_i'=G_i p + d(G_i p)-d(p), so the first plane stays fixed.
Local displacements compare x_i' and x_(i-1)' in the previous rigid image basis.
This explicit adaptation reduces exactly to the official rigid definition at d=0.
"""
    tool_global: torch.Tensor
    pixel_to_mm: torch.Tensor
    image_mm_to_tool: torch.Tensor
    field: torch.Tensor | None = None
    bounds: tuple | None = None

    @cached_property
    def _image_global(self):
        return torch.linalg.inv(self.image_mm_to_tool) @ self.tool_global @ self.image_mm_to_tool

    def image_global(self):
        return self._image_global

    def positions(self, index, points):
        base = self.pixel_to_mm @ points
        result = (self.image_global()[index] @ base)[:3]
        if self.field is not None:
            from baselines.nr_rec_fus.models.volume import sample_field
            delta = sample_field(self.field, result[None, None].to(self.field), self.bounds)
            anchor = sample_field(self.field, base[:3][None, None].to(self.field), self.bounds)
            result = result + (delta - anchor)[0, 0].to(result)
        return result

    def displacement(self, index, points, local=False):
        current = self.positions(index, points)
        if not local:
            return current - (self.pixel_to_mm @ points)[:3]
        previous = self.positions(index - 1, points)
        rotation_inv = torch.linalg.inv(self.image_global()[index - 1])[:3, :3]
        return rotation_inv @ (current - previous)


def landmark_points(landmarks, n, *, device='cpu', dtype=torch.float64):
    value = torch.as_tensor(landmarks, device=device, dtype=dtype)
    if value.ndim != 2 or value.shape[1] != 3 or not len(value) or not torch.isfinite(value).all():
        raise ValueError('landmarks must be a finite nonempty [L,3] array')
    ids = value[:, 0].long()
    if not torch.equal(ids.to(value), value[:, 0]) or (ids < 1).any() or (ids >= n).any():
        raise ValueError('Official landmark frame indices must be integers in [1,N-1]')
    points = torch.cat((value[:, 1:].T, torch.zeros(1, len(value), device=device, dtype=dtype),
                        torch.ones(1, len(value), device=device, dtype=dtype)))
    return ids, points


@torch.no_grad()
def landmark_ddfs(prediction, landmarks):
    ids, points = landmark_points(landmarks, len(prediction.tool_global), device=prediction.tool_global.device,
                                 dtype=prediction.tool_global.dtype)
    return tuple(torch.cat([prediction.displacement(int(i), points[:, k:k+1], local)
                            for k, i in enumerate(ids)], 1) for local in (False, True))


@torch.no_grad()
def iter_pixel_ddfs(prediction, height=480, width=640, chunk_size=16384):
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    points = image_points(height, width, device=prediction.tool_global.device, dtype=prediction.tool_global.dtype)
    transforms = prediction.image_global()
    inverse_rotations = torch.linalg.inv(transforms)[:, :3, :3]
    # Traverse frames inside each spatial chunk: reuse the previous frame's
    # positions and the reference-plane deformation (one field lookup per frame).
    for start in range(0, points.shape[1], chunk_size):
        p = points[:, start:start + chunk_size]
        base = prediction.pixel_to_mm @ p
        previous = prediction.positions(0, p)
        anchor = None
        if prediction.field is not None:
            from baselines.nr_rec_fus.models.volume import sample_field
            anchor = sample_field(prediction.field, base[:3][None, None].to(prediction.field), prediction.bounds)
        for i in range(1, len(prediction.tool_global)):
            current = (transforms[i] @ base)[:3]
            if prediction.field is not None:
                delta = sample_field(prediction.field, current[None, None].to(prediction.field), prediction.bounds)
                current = current + (delta - anchor)[0, 0].to(current)
            yield i - 1, start, current - base[:3], inverse_rotations[i - 1] @ (current - previous)
            previous = current


@torch.no_grad()
def score_scan(prediction, tool_to_world, landmarks, height=480, width=640, chunk_size=16384):
    # Labels enter only the scorer, after inference has completed.
    labels = torch.as_tensor(tool_to_world).to(prediction.tool_global)
    if labels.shape != prediction.tool_global.shape or not torch.isfinite(labels).all():
        raise ValueError('Labels must be finite [N,4,4] and match the predicted scan length')
    truth = ScanPrediction(global_from_labels(labels),
                           prediction.pixel_to_mm, prediction.image_mm_to_tool)
    sums = np.zeros(2, dtype=np.float64)
    count = 0
    for pred, target in zip(iter_pixel_ddfs(prediction, height, width, chunk_size),
                            iter_pixel_ddfs(truth, height, width, chunk_size)):
        for j in range(2):
            sums[j] += torch.linalg.vector_norm(pred[2+j] - target[2+j], dim=0).sum().item()
        count += pred[2].shape[1]
    if count == 0:
        raise ValueError('A scan needs at least two frames')
    pred_l, true_l = landmark_ddfs(prediction, landmarks), landmark_ddfs(truth, landmarks)
    distances = [torch.linalg.vector_norm(p - t, dim=0).mean().item() for p, t in zip(pred_l, true_l)]
    return {'GPE': sums[0] / count, 'GLE': distances[0], 'LPE': sums[1] / count, 'LLE': distances[1]}
