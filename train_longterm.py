#!/usr/bin/env python3
"""Long-Term Dependency training with fixed validation and resumable state."""
import argparse
from pathlib import Path
import numpy as np
import torch
from datasets.longterm import LongTermDataset
from models_longterm import (LongTermEfficientNet, reference_image_points,
                            relative_tool_transforms, transform_points, euler_xyz_to_matrix)
from training import seed_all, select_device, run_training


def loss_and_metric(model, batch, image_to_tool, points, device):
    frames, tforms = batch['frames'].to(device), batch['tforms'].to(device)
    target = transform_points(relative_tool_transforms(tforms), image_to_tool, points)
    pred = transform_points(euler_xyz_to_matrix(model(frames)), image_to_tool, points)
    loss = (pred - target).square().mean()
    distance = torch.linalg.vector_norm(pred - target, dim=2).mean()
    return loss, distance


def main(args):
    seed_all(args.seed)
    device = select_device(args.device)
    pre = Path(args.preprocessed)
    image_to_tool = torch.from_numpy(np.load(pre / 'image_to_tool_downsampled.npy')).float().to(device)
    # Preserve the published adaptation's four-corner training convention so
    # existing weights remain usable. Full-scan evaluation uses official pixels.
    points = reference_image_points().to(device)
    train = LongTermDataset(args.root, 'train', pre, args.num_samples, args.sample_range,
                            seed=args.seed, val_windows=args.val_windows)
    val = LongTermDataset(args.root, 'val', pre, args.num_samples, args.sample_range,
                          seed=args.seed, val_windows=args.val_windows)
    model = LongTermEfficientNet(num_samples=args.num_samples).to(device)

    def objective(model, batch):
        loss, distance = loss_and_metric(model, batch, image_to_tool, points, device)
        return loss, {'distance_mm': distance}

    config = {**vars(args), 'method': 'longterm', 'height': 120, 'width': 160,
              'model': 'efficientnet_b1', 'point_convention': 'legacy_zero_based_four_corners'}
    run_training(model, train, val, objective, config, device, args.output, args.epochs,
                 args.batch_size, args.lr, args.workers, args.resume, args.max_steps, args.val_steps)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='data/tus-rec2024')
    p.add_argument('--preprocessed', default='data/tus-rec2024/preprocessed/longterm')
    p.add_argument('--output', default='runs/longterm_v2')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--max-steps', type=int, default=0, help='Debug training batches per epoch; 0 means all')
    p.add_argument('--val-steps', type=int, default=0, help='Debug validation batches; 0 means all')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--num-samples', type=int, default=10)
    p.add_argument('--sample-range', type=int, default=10)
    p.add_argument('--val-windows', type=int, default=3)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--resume', default='')
    p.add_argument('--device', default='cuda')
    main(p.parse_args())
