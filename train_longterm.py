#!/usr/bin/env python3
"""Reproduce the Long-Term Dependency pose-regression baseline on TUS-REC2024."""
import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.longterm import LongTermDataset, collate_longterm
from models_longterm import LongTermEfficientNet, pair_samples, reference_image_points, relative_tool_transforms, transform_points, euler_xyz_to_matrix


def loss_and_metric(model, batch, image_to_tool, points, device):
    frames, tforms = batch['frames'].to(device), batch['tforms'].to(device)
    target = transform_points(relative_tool_transforms(tforms), image_to_tool, points)
    pred = transform_points(euler_xyz_to_matrix(model(frames)), image_to_tool, points)
    # Published implementation uses MSE point supervision and reports Euclidean point distance.
    loss = (pred - target).pow(2).mean()
    distance = (pred - target).pow(2).sum(dim=2).sqrt().mean()
    return loss, distance


def evaluate(model, loader, image_to_tool, points, device):
    model.eval(); losses = []; distances = []
    with torch.no_grad():
        for batch in loader:
            loss, distance = loss_and_metric(model, batch, image_to_tool, points, device)
            losses.append(loss.item()); distances.append(distance.item())
    return float(np.mean(losses)), float(np.mean(distances))


def main(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    pre = Path(args.preprocessed)
    image_to_tool = torch.from_numpy(np.load(pre / 'image_to_tool_downsampled.npy')).float().to(device)
    points = reference_image_points().to(device)
    train = LongTermDataset(args.root, 'train', pre, args.num_samples, args.sample_range)
    val = LongTermDataset(args.root, 'val', pre, args.num_samples, args.sample_range)
    train_loader = DataLoader(train, args.batch_size, shuffle=True, num_workers=args.workers,
                              collate_fn=collate_longterm, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val, 1, shuffle=False, num_workers=args.workers,
                            collate_fn=collate_longterm, persistent_workers=args.workers > 0)
    model = LongTermEfficientNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    best = float('inf'); history = []; start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        if checkpoint.get('optimizer'):
            optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = int(checkpoint.get('epoch', -1)) + 1
        history_path = output / 'history.json'
        if history_path.exists():
            history = json.loads(history_path.read_text())
            if history:
                best = min(float(x['val_distance_mm']) for x in history)
        print(f'Resuming from epoch {start_epoch}', flush=True)
    for epoch in range(start_epoch, args.epochs):
        model.train(); train_losses = []; train_distances = []
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            loss, distance = loss_and_metric(model, batch, image_to_tool, points, device)
            loss.backward(); optimizer.step()
            train_losses.append(loss.item()); train_distances.append(distance.item())
            if args.max_steps and step + 1 >= args.max_steps: break
        val_loss, val_distance = evaluate(model, val_loader, image_to_tool, points, device)
        row = {'epoch': epoch, 'train_loss': float(np.mean(train_losses)),
               'train_distance_mm': float(np.mean(train_distances)),
               'val_loss': val_loss, 'val_distance_mm': val_distance}
        history.append(row); print(json.dumps(row), flush=True)
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'args': vars(args), 'epoch': epoch}, output / 'last.pt')
        if val_distance < best:
            best = val_distance; torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                             'args': vars(args), 'epoch': epoch}, output / 'best.pt')
        (output / 'history.json').write_text(json.dumps(history, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='data/tus-rec2024'); p.add_argument('--preprocessed', default='data/tus-rec2024/preprocessed/longterm')
    p.add_argument('--output', default='runs/longterm'); p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--max-steps', type=int, default=0, help='Debug cap per epoch; 0 uses all 1200 scans')
    p.add_argument('--batch-size', type=int, default=32); p.add_argument('--workers', type=int, default=4)
    p.add_argument('--num-samples', type=int, default=10); p.add_argument('--sample-range', type=int, default=10)
    p.add_argument('--lr', type=float, default=1e-4); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--resume', default='', help='Checkpoint path, e.g. runs/longterm/last.pt')
    p.add_argument('--device', default='cuda'); main(p.parse_args())
