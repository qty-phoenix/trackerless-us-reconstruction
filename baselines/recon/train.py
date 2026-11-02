#!/usr/bin/env python3
"""Two-stage RecON training on the common TUS-REC2024 protocol."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F

from geometry import calibration_tensors, image_points
from training import atomic_save, run_training, seed_all, select_device
from baselines.nr_rec_fus.models.volume import splat, volume_bounds
from .dataset import RecONDataset
from .models import (RecON, chain_relative, correlation_loss, dof_to_matrix,
                     matrix_to_dof, pair_inputs, scaled_motion, unscaled_motion)


def motion_labels(tforms, spatial):
    relative_tool = torch.linalg.inv(tforms[:, :-1]) @ tforms[:, 1:]
    relative_image = torch.linalg.inv(spatial) @ relative_tool @ spatial
    return matrix_to_dof(relative_image)


def supervised_losses(model, frames, target, config, include_motion=True, inputs=None):
    inputs = pair_inputs(frames) if inputs is None else inputs
    prediction, feature = model.backbone(inputs, return_feature=include_motion)
    target_scaled = scaled_motion(target, config['angle_scale'])
    loss_dist = F.l1_loss(prediction[..., :3], target_scaled[..., :3]) * 3
    loss_angle = F.l1_loss(prediction[..., 3:], target_scaled[..., 3:]) * 3
    loss_corr = correlation_loss(target_scaled, prediction)
    losses = {'loss_dist': loss_dist, 'loss_angle': loss_angle, 'loss_corr': loss_corr}
    if include_motion:
        predicted_motion = prediction[..., :3].norm(dim=-1).add(1e-6)
        loss_motion = (feature.norm(dim=-1) / predicted_motion).mean() * config['motion_weight']
        losses['loss_motion'] = loss_motion
    return prediction, losses


def backbone_objective(model, batch, scale, spatial, config, device):
    frames, tforms = batch['frames'].to(device), batch['tforms'].to(device)
    inputs = batch['inputs'].to(device)
    target = motion_labels(tforms, spatial)
    prediction, losses = supervised_losses(model, frames, target, config, inputs=inputs)
    predicted_matrix = dof_to_matrix(unscaled_motion(prediction, config['angle_scale']))
    target_matrix = dof_to_matrix(target)
    points = image_points(*frames.shape[-2:], device=device, dtype=frames.dtype)
    base = scale @ points
    predicted_points = (predicted_matrix @ base)[:, :, :3]
    target_points = (target_matrix @ base)[:, :, :3]
    distance = torch.linalg.vector_norm(predicted_points - target_points, dim=2).mean()
    corners = torch.tensor([0, frames.shape[-1] - 1,
                            (frames.shape[-2] - 1) * frames.shape[-1],
                            frames.shape[-2] * frames.shape[-1] - 1], device=device)
    selection = torch.linalg.vector_norm(
        predicted_points[:, 0].index_select(-1, corners) -
        target_points[:, 0].index_select(-1, corners), dim=1).mean()
    return sum(losses.values()), {'distance_mm': distance,
                                  'selection_distance_mm': selection, **losses}


def reconstruction_pair(model, frames, inputs, target, scale, config):
    prediction, _ = model.backbone(inputs)
    predicted_global = chain_relative(dof_to_matrix(unscaled_motion(
        prediction, config['angle_scale'])))
    target_global = chain_relative(dof_to_matrix(target))
    size = config['reconstruction_height'], config['reconstruction_width']
    small_frames = F.interpolate(frames, size=size, mode='bilinear', align_corners=False)
    points = scale @ image_points(*size, device=frames.device, dtype=frames.dtype)
    predicted_points = (predicted_global @ points)[:, :, :3]
    target_points = (target_global @ points)[:, :, :3]
    predicted_bounds = volume_bounds(predicted_points, config['margin_mm'])
    target_bounds = volume_bounds(target_points, config['margin_mm'])
    fake, _ = splat(predicted_points, small_frames, predicted_bounds, config['volume_shape'])
    real, _ = splat(target_points, small_frames, target_bounds, config['volume_shape'])
    return real, fake


def discriminator_objective(model, batch, scale, spatial, config, device):
    training = model.training
    model.backbone.eval()
    model.discriminator.train(training)
    frames, tforms = batch['frames'].to(device), batch['tforms'].to(device)
    inputs = batch['inputs'].to(device)
    target = motion_labels(tforms, spatial)
    with torch.no_grad():
        real, fake = reconstruction_pair(model, frames, inputs, target, scale, config)
    real_logits, fake_logits = model.discriminator(real), model.discriminator(fake)
    real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    loss = (real_loss + fake_loss) / 2
    accuracy = ((real_logits >= 0).float().mean() + (fake_logits < 0).float().mean()) / 2
    return loss, {'discriminator_accuracy': accuracy, 'real_loss': real_loss, 'fake_loss': fake_loss}


def build_online_bank(dataset, spatial, config):
    count = min(config['pseudo_bank_size'], len(dataset))
    indices = torch.linspace(0, len(dataset) - 1, count).round().long().unique().tolist()
    frames, targets, metadata = [], [], []
    dataset.set_epoch(0)
    for index in indices:
        sample = dataset[index]
        value = sample['frames']
        target = motion_labels(sample['tforms'][None].to(spatial), spatial)[0]
        frames.append(value.mul(255).round().clamp(0, 255).to(torch.uint8))
        targets.append(target.to(torch.float16).cpu())
        metadata.append({'scan_id': sample['scan_id'], 'subject': sample['subject'],
                         'frame_indices': sample['indices'].tolist()})
    subjects = sorted({item['subject'] for item in metadata})
    return {'frames_uint8': torch.stack(frames), 'target_dof': torch.stack(targets),
            'metadata': metadata, 'source_split': 'train', 'subjects': subjects}


def attach_online_bank(output, bank):
    for name in ('best.pt', 'last.pt'):
        path = Path(output) / name
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        checkpoint['online_bank'] = bank
        atomic_save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='baselines/recon/configs/tus_rec2024.json')
    parser.add_argument('--stage', choices=('backbone', 'discriminator'), default='backbone')
    parser.add_argument('--backbone-checkpoint', default='')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output')
    parser.add_argument('--resume', default='')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--steps', '--max-steps', dest='max_steps', type=int, default=0)
    parser.add_argument('--val-steps', type=int, default=0)
    parser.add_argument('--no-imagenet-pretrained', action='store_true')
    parser.add_argument('--pseudo-bank-size', type=int,
                        help='Debug storage override; formal protocol uses all 1200 train scans')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    config.update(method='recon', stage=args.stage)
    if args.no_imagenet_pretrained:
        config['imagenet_pretrained'] = False
    if args.pseudo_bank_size is not None:
        if args.pseudo_bank_size < 1:
            parser.error('--pseudo-bank-size must be positive')
        config['pseudo_bank_size'] = args.pseudo_bank_size
    default_epochs = config[f'{args.stage}_epochs']
    config['epochs'] = args.epochs if args.epochs is not None else default_epochs
    config['selection_metric'] = (config.get('selection_metric', 'val_distance_mm')
                                  if args.stage == 'backbone' else 'val_loss')
    output = args.output or (config['backbone_output'] if args.stage == 'backbone' else config['output'])
    seed_all(config['seed'])
    device = select_device(args.device)
    torch.hub.set_dir(config['torch_hub_dir'])
    size = config['height'], config['width']
    scale, spatial = calibration_tensors(Path(config['root']) / 'calib_matrix.csv', device, size=size)
    reco_scale, _ = calibration_tensors(Path(config['root']) / 'calib_matrix.csv', device,
                                        size=(config['reconstruction_height'], config['reconstruction_width']))
    datasets = [RecONDataset(config['root'], split, config['preprocessed'], config['num_samples'],
                             config['sample_range'], size, config['seed'], config['val_windows'],
                             config.get('frame_cache'))
                for split in ('train', 'val')]
    pretrained = config['imagenet_pretrained'] and args.stage == 'backbone' and not args.resume
    model = RecON(imagenet_pretrained=pretrained).to(device)
    if args.stage == 'discriminator':
        if not args.resume:
            if not args.backbone_checkpoint:
                parser.error('--backbone-checkpoint is required for a new discriminator run')
            source = torch.load(args.backbone_checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(source['model'])
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
        objective = lambda m, b: discriminator_objective(m, b, reco_scale, spatial, config, device)
    else:
        for parameter in model.discriminator.parameters():
            parameter.requires_grad_(False)
        objective = lambda m, b: backbone_objective(m, b, scale, spatial, config, device)
    run_training(model, *datasets, objective, config, device, output, config['epochs'],
                 config['batch_size'], config['lr'], args.workers, args.resume,
                 args.max_steps, args.val_steps, config['weight_decay'])
    if args.stage == 'discriminator':
        # Embed compressed source frames and train-only labels for PSC.  Runtime
        # inference therefore never opens validation/test pose files.
        bank_dataset = RecONDataset(config['root'], 'train', config['preprocessed'],
                                    config['num_samples'], config['sample_range'], size,
                                    config['seed'], config['val_windows'],
                                    config.get('frame_cache'), with_inputs=False)
        try:
            bank = build_online_bank(bank_dataset, spatial, config)
        finally:
            bank_dataset.close()
        attach_online_bank(output, bank)


if __name__ == '__main__':
    main()
