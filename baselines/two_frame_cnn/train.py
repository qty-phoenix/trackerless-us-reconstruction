"""Train the Prevost et al. 2018 two-frame CNN on TUS-REC2024."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F

from geometry import calibration_tensors, image_points
from training import run_training, seed_all, select_device
from baselines.recon.models import dof_to_matrix, matrix_to_dof
from .dataset import TwoFrameDataset
from .models import TwoFrameCNN


def motion_labels(tforms, spatial, scale, flipped, width=128):
    """Adjacent image transforms as translation (mm) plus XYZ Euler (degrees)."""
    relative_tool = torch.linalg.inv(tforms[:, 0]) @ tforms[:, 1]
    relative_image = torch.linalg.inv(spatial) @ relative_tool @ spatial
    if flipped.any():
        # A horizontal flip maps resized pixel x to W+1-x.  Express the
        # reflection in image millimetres and conjugate the relative pose.
        reflection = torch.eye(4, device=tforms.device, dtype=tforms.dtype)
        reflection[0, 0] = -1
        reflection[0, 3] = scale[0, 0] * (width + 1)
        transformed = reflection @ relative_image @ reflection
        relative_image = torch.where(flipped[:, None, None], transformed, relative_image)
    dof = matrix_to_dof(relative_image)
    return torch.cat((dof[:, :3], torch.rad2deg(dof[:, 3:])), -1)


def objective(model, batch, scale, spatial, config, device):
    frames = batch['frames'].to(device)
    tforms = batch['tforms'].to(device)
    flipped = batch['flipped'].to(device)
    target = motion_labels(tforms, spatial, scale, flipped, config['width'])
    prediction = model(frames)
    loss = F.mse_loss(prediction, target)
    with torch.no_grad():
        predicted_dof = torch.cat((prediction[:, :3], torch.deg2rad(prediction[:, 3:])), -1)
        target_dof = torch.cat((target[:, :3], torch.deg2rad(target[:, 3:])), -1)
        predicted_matrix = dof_to_matrix(predicted_dof)
        target_matrix = dof_to_matrix(target_dof)
        points = scale @ image_points(*frames.shape[-2:], device=device, dtype=frames.dtype)
        predicted_points = (predicted_matrix @ points)[:, :3]
        target_points = (target_matrix @ points)[:, :3]
        distance = torch.linalg.vector_norm(predicted_points - target_points, dim=1).mean()
        corners = torch.tensor([0, frames.shape[-1] - 1,
                                (frames.shape[-2] - 1) * frames.shape[-1],
                                frames.shape[-2] * frames.shape[-1] - 1], device=device)
        selection = torch.linalg.vector_norm(
            predicted_points.index_select(-1, corners) -
            target_points.index_select(-1, corners), dim=1).mean()
    return loss, {'distance_mm': distance,
                  'selection_distance_mm': selection,
                  'translation_mae': (prediction[:, :3] - target[:, :3]).abs().mean(),
                  'rotation_mae_deg': (prediction[:, 3:] - target[:, 3:]).abs().mean(),
                  'motion_mae': (prediction - target).abs().mean()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='baselines/two_frame_cnn/configs/tus_rec2024.json')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output')
    parser.add_argument('--resume', default='')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--steps', '--max-steps', dest='max_steps', type=int, default=0)
    parser.add_argument('--val-steps', type=int, default=0)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.epochs is not None:
        config['epochs'] = args.epochs
    config.update(method='two_frame_cnn')
    config.setdefault('selection_metric', 'val_distance_mm')
    seed_all(config['seed'])
    device = select_device(args.device)
    size = config['height'], config['width']
    scale, spatial = calibration_tensors(Path(config['root']) / 'calib_matrix.csv',
                                         device, size=size)
    train = TwoFrameDataset(config['root'], 'train', config['preprocessed'], size, config['seed'],
                            config['val_windows'], config['flip_h_prob'],
                            config['reverse_sweep_prob'], config['skip_frame_prob'],
                            config.get('frame_cache'), config['pairs_per_scan'])
    val = TwoFrameDataset(config['root'], 'val', config['preprocessed'], size, config['seed'],
                          config['val_windows'], 0, 0, 0, config.get('frame_cache'))
    model = TwoFrameCNN(config['dropout'], config['height'], config['width']).to(device)
    fn = lambda m, b: objective(m, b, scale, spatial, config, device)
    run_training(model, train, val, fn, config, device,
                 args.output or config['output'], config['epochs'], config['batch_size'],
                 config['lr'], args.workers, args.resume, args.max_steps, args.val_steps,
                 config['weight_decay'])


if __name__ == '__main__':
    main()
