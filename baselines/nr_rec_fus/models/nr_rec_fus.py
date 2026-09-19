import torch
from torch import nn
from .rigid_pose import RigidPoseNet, se3_from_vec
from .deformation import DeformationNet
from .volume import volume_bounds, splat, anchor_points

class NRRecFUS(nn.Module):
    def __init__(self, num_samples=4, model='efficientnet_b0', volume_shape=(32, 64, 64),
                 deform_channels=8, max_displacement_mm=5., margin_mm=10.):
        super().__init__()
        if num_samples < 2 or min(volume_shape) < 8 or margin_mm <= 0:
            raise ValueError('Require >=2 frames, volume dimensions >=8 and positive margin')
        self.num_samples, self.volume_shape = num_samples, tuple(volume_shape)
        self.margin_mm = margin_mm
        self.rigid = RigidPoseNet(num_samples, model=model)
        self.deform = DeformationNet(deform_channels, max_displacement_mm)

    def forward(self, frames, pixel_to_mm, image_mm_to_tool):
        # No labels enter this forward pass or the volume's coordinate bounds.
        from geometry import image_points
        pose_vec = self.rigid(frames)
        pose = se3_from_vec(pose_vec)
        identity = torch.eye(4, device=frames.device, dtype=frames.dtype)[None, None].expand(frames.shape[0], 1, 4, 4)
        transforms = torch.cat((identity, pose), 1)
        points = image_points(*frames.shape[-2:], device=frames.device, dtype=frames.dtype)
        base = (pixel_to_mm @ points)[:3][None, None].expand(frames.shape[0], 1, -1, -1)
        image_transforms = torch.linalg.inv(image_mm_to_tool) @ transforms @ image_mm_to_tool
        rigid_points = (image_transforms @ (pixel_to_mm @ points))[:, :, :3]
        bounds = volume_bounds(rigid_points, self.margin_mm)
        volume, occupancy = splat(rigid_points, frames, bounds, self.volume_shape)
        field = self.deform(volume)
        refined = anchor_points(rigid_points, base, field, bounds)
        return {'pose_vec': pose_vec, 'pose': pose, 'rigid_points': rigid_points,
                'points': refined, 'volume': volume, 'occupancy': occupancy,
                'field': field, 'bounds': bounds}
