"""Train the paper-defined DC2-Net losses on the TUS-REC2024 split."""
import argparse
import json
import math
from pathlib import Path
import torch
from torch.nn import functional as F

from geometry import calibration_tensors, image_points
from training import run_training, seed_all, select_device
from baselines.recon.models import dof_to_matrix, matrix_to_dof
from .dataset import DC2Dataset
from .models import DC2Net


def motion_labels(tforms, spatial):
    """Mean neighboring motion for every video segment, in mm and degrees."""
    relative_tool = torch.linalg.inv(tforms[..., :-1, :, :]) @ tforms[..., 1:, :, :]
    relative_image = torch.linalg.inv(spatial) @ relative_tool @ spatial
    dof = matrix_to_dof(relative_image)
    dof = torch.cat((dof[..., :3], torch.rad2deg(dof[..., 3:])), -1)
    return dof.mean(-2)


def pearson_case_loss(prediction, target, eps=1e-6):
    """Pearson loss over the K segments of each scan and every DoF."""
    x = prediction - prediction.mean(1, keepdim=True)
    y = target - target.mean(1, keepdim=True)
    numerator = (x * y).sum(1)
    denominator = x.square().sum(1).sqrt() * y.square().sum(1).sqrt()
    valid = denominator > eps
    correlation = numerator / denominator.clamp_min(eps)
    # Constant target axes contain no correlation information.
    return 1 - correlation[valid].mean() if valid.any() else prediction.new_zeros(())


def margin_ranking_loss(embedding, target, margin):
    """All valid within-batch anchor/pair rankings from the paper's loss."""
    feature_distance = torch.cdist(embedding, embedding)
    label_distance = torch.cdist(target, target)
    terms = []
    count = len(embedding)
    for anchor in range(count):
        others = [i for i in range(count) if i != anchor]
        for left_index, left in enumerate(others):
            for right in others[left_index + 1:]:
                coefficient = torch.sign(label_distance[anchor, right] -
                                         label_distance[anchor, left])
                if coefficient != 0:
                    terms.append(F.relu(coefficient *
                        (feature_distance[anchor, left] - feature_distance[anchor, right]) + margin))
    return torch.stack(terms).mean() if terms else embedding.new_zeros(())


def objective(model, batch, scale, spatial, config, device):
    frames = batch['frames'].to(device)
    tforms = batch['tforms'].to(device)
    batch_size, cases, samples, height, width = frames.shape
    target = motion_labels(tforms, spatial)
    motion_scale = torch.as_tensor(config['motion_scale'], device=device,
                                   dtype=frames.dtype)
    prediction_scaled, embedding, _ = model(frames.reshape(-1, samples, height, width))
    prediction = (prediction_scaled * motion_scale).reshape(batch_size, cases, 6)
    embedding = embedding.reshape(batch_size, cases, -1)
    mse = F.mse_loss(prediction, target)
    correlation = pearson_case_loss(prediction, target)
    ranking = margin_ranking_loss(embedding.flatten(0, 1), target.flatten(0, 1),
                                  config['margin'])
    loss = (config['mse_weight'] * mse + config['correlation_weight'] * correlation +
            config['ranking_weight'] * ranking)
    with torch.no_grad():
        predicted_dof = torch.cat((prediction[..., :3],
                                   torch.deg2rad(prediction[..., 3:])), -1)
        target_dof = torch.cat((target[..., :3], torch.deg2rad(target[..., 3:])), -1)
        predicted_matrix = dof_to_matrix(predicted_dof)
        target_matrix = dof_to_matrix(target_dof)
        points = scale @ image_points(*frames.shape[-2:], device=device, dtype=frames.dtype)
        predicted_points = (predicted_matrix @ points)[..., :3, :]
        target_points = (target_matrix @ points)[..., :3, :]
        distance = torch.linalg.vector_norm(predicted_points - target_points, dim=-2).mean()
        corners = torch.tensor([0, frames.shape[-1] - 1,
                                (frames.shape[-2] - 1) * frames.shape[-1],
                                frames.shape[-2] * frames.shape[-1] - 1], device=device)
        selection = torch.linalg.vector_norm(
            predicted_points[:, 0].index_select(-1, corners) -
            target_points[:, 0].index_select(-1, corners), dim=1).mean()
    return loss, {'distance_mm': distance, 'MSE': mse, 'COR': correlation,
                  'selection_distance_mm': selection,
                  'MARGIN': ranking, 'motion_mae': (prediction - target).abs().mean()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='baselines/dc2_net/configs/tus_rec2024.json')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output')
    parser.add_argument('--resume', default='')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--steps', '--max-steps', dest='max_steps', type=int, default=0)
    parser.add_argument('--val-steps', type=int, default=0)
    parser.add_argument('--tiny-model', action='store_true',
                        help='Debug-only shallow model; requires a limited-step run')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.tiny_model:
        if not args.max_steps or not args.val_steps:
            parser.error('--tiny-model is restricted to capped debug runs')
        config.update(model_layers=[1, 1, 1, 1], groups=1, base_width=4)
    config.update(method='dc2_net')
    config.setdefault('selection_metric', 'val_distance_mm')
    seed_all(config['seed'])
    device = select_device(args.device)
    size = config['height'], config['width']
    scale, spatial = calibration_tensors(Path(config['root']) / 'calib_matrix.csv',
                                         device, size=size)
    datasets = [DC2Dataset(config['root'], split, config['preprocessed'],
                           config['num_samples'], config['case_windows'], size,
                           config['seed'], config['val_windows'], config.get('frame_cache'))
                for split in ('train', 'val')]
    model = DC2Net(tuple(config['model_layers']), config['groups'],
                   config['base_width']).to(device)
    loss_fn = lambda m, b: objective(m, b, scale, spatial, config, device)
    run_training(model, *datasets, loss_fn, config, device,
                 args.output or config['output'], config['epochs'], config['batch_size'],
                 config['lr'], args.workers, args.resume, args.max_steps, args.val_steps,
                 config['weight_decay'])


if __name__ == '__main__':
    main()
