"""NR-Rec-FUS-inspired, single-input 3D registration adaptation for TUS-REC2024."""
import argparse
import json
from pathlib import Path
import torch
from geometry import calibration_tensors, image_points
from training import seed_all, select_device, run_training
from .datasets.nr_rec_dataset import NRRecDataset
from .models.nr_rec_fus import NRRecFUS
from .models.volume import splat, warp_volume, bending_energy


def build_model(config):
    return NRRecFUS(**{k: config[k] for k in ('num_samples', 'model', 'volume_shape',
                       'deform_channels', 'max_displacement_mm', 'margin_mm') if k in config})


def loss_and_metrics(model, batch, scale, spatial, config, device):
    frames, tforms = batch['frames'].to(device), batch['tforms'].to(device)
    out = model(frames, scale, spatial)
    relative = torch.linalg.inv(tforms[:, :1]) @ tforms
    image_transforms = torch.linalg.inv(spatial) @ relative @ spatial
    points = image_points(*frames.shape[-2:], device=device, dtype=frames.dtype)
    target = (image_transforms @ (scale @ points))[:, :, :3]
    rigid_loss = (out['rigid_points'][:, 1:] - target[:, 1:]).square().mean()
    refined_loss = (out['points'][:, 1:] - target[:, 1:]).square().mean()
    # Labels are used only in losses. Model input and grid bounds are predicted.
    target_volume, occupancy = splat(target, frames, out['bounds'], model.volume_shape)
    warped = warp_volume(target_volume, out['field'], out['bounds'])
    support = ((occupancy > 0) | (out['occupancy'] > 0)).to(frames.dtype)
    photo = ((warped - out['volume']).square() * support).sum() / support.sum().clamp_min(1)
    smooth = bending_energy(out['field'], out['bounds'])
    registration = photo + config['smooth_weight'] * smooth
    loss = rigid_loss + config['refined_loss_weight'] * refined_loss + config['reg_loss_weight'] * registration
    return loss, {'distance_mm': torch.linalg.vector_norm(out['points'][:, 1:] - target[:, 1:], dim=2).mean(),
                  'rigid_distance_mm': torch.linalg.vector_norm(out['rigid_points'][:, 1:] - target[:, 1:], dim=2).mean(),
                  'rigid_loss': rigid_loss, 'refined_loss': refined_loss,
                  'registration_loss': registration, 'photo_loss': photo, 'bending_loss': smooth}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='baselines/nr_rec_fus/configs/tus_rec2024.json')
    p.add_argument('--device', default='cuda')
    p.add_argument('--output', default=None)
    p.add_argument('--resume', default='')
    p.add_argument('--epochs', type=int)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--steps', '--max-steps', dest='max_steps', type=int, default=0,
                   help='Debug training batches per epoch; still validates and saves')
    p.add_argument('--val-steps', type=int, default=0)
    a = p.parse_args()
    c = json.loads(Path(a.config).read_text())
    allowed = {'root', 'preprocessed', 'num_samples', 'sample_range', 'height', 'width',
               'batch_size', 'epochs', 'lr', 'reg_loss_weight', 'smooth_weight', 'model',
               'output', 'seed', 'val_windows', 'volume_shape', 'deform_channels',
               'max_displacement_mm', 'margin_mm', 'refined_loss_weight'}
    if set(c) - allowed:
        raise ValueError(f'Unknown configuration keys: {set(c) - allowed}')
    c = {'seed': 0, 'val_windows': 3, 'refined_loss_weight': 1., **c}
    c['method'] = 'nr_rec_fus'
    if a.epochs is not None:
        c['epochs'] = a.epochs
    if min(c['reg_loss_weight'], c['smooth_weight'], c['refined_loss_weight']) < 0:
        raise ValueError('Loss weights must be nonnegative')
    seed_all(c['seed'])
    device = select_device(a.device)
    size = c['height'], c['width']
    scale, spatial = calibration_tensors(Path(c['root']) / 'calib_matrix.csv', device, size=size)
    datasets = [NRRecDataset(c['root'], split, c['preprocessed'], c['num_samples'], c['sample_range'],
                             size, c['seed'], c['val_windows']) for split in ('train', 'val')]
    model = build_model(c).to(device)

    def objective(model, batch):
        return loss_and_metrics(model, batch, scale, spatial, c, device)

    run_training(model, *datasets, objective, c, device, a.output or c.get('output', 'runs/nr_rec_fus'),
                 c['epochs'], c['batch_size'], c['lr'], a.workers, a.resume, a.max_steps, a.val_steps)


if __name__ == '__main__':
    main()
