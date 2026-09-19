"""Shared reproducible training, validation, history and resumable checkpoints."""
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device(name):
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; explicitly use --device cpu for a smoke test')
    return device


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def run_training(model, train, val, loss_fn, config, device, output, epochs,
                 batch_size, lr, workers=0, resume='', max_steps=0, val_steps=0):
    if epochs < 1 or batch_size < 1 or min(max_steps, val_steps, workers) < 0:
        raise ValueError('Invalid training limits')
    train_subjects = {r['subject'] for r in train.rows}
    val_subjects = {r['subject'] for r in val.rows}
    if train_subjects & val_subjects:
        raise ValueError('Train and validation subjects overlap')
    output = Path(output)
    if not resume and any((output / f).exists() for f in ('best.pt', 'last.pt', 'history.json')):
        raise FileExistsError(f'{output} has a previous run; use --resume or a new --output')
    output.mkdir(parents=True, exist_ok=True)
    config = {**config, 'max_steps': max_steps, 'val_steps': val_steps, 'batch_size': batch_size}
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start, best, history = 0, float('inf'), []
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        old = checkpoint.get('config', checkpoint.get('args', {}))
        for key in ('method', 'num_samples', 'sample_range', 'height', 'width', 'model',
                    'volume_shape', 'deform_channels', 'max_displacement_mm', 'margin_mm',
                    'reg_loss_weight', 'smooth_weight', 'refined_loss_weight', 'val_windows', 'seed',
                    'max_steps', 'val_steps', 'batch_size'):
            if key in old and key in config and old[key] != config[key]:
                raise ValueError(f'Resume configuration mismatch: {key}')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start = checkpoint['epoch'] + 1
        history = checkpoint.get('history', [])
        # Old Long-Term checkpoint selection used random validation windows. Reset
        # best when migrating it to the new fixed validation protocol.
        best = checkpoint.get('best', float('inf')) if checkpoint.get('version') == 2 else float('inf')
        if Path(resume).resolve().parent != output.resolve() and checkpoint.get('version') == 2:
            # A new output directory is a new selection run. Do not retain an
            # inaccessible historical best value without its corresponding weights.
            best = float('inf')
        if 'rng' in checkpoint:
            restore_rng(checkpoint['rng'])
        print(f'Resuming at epoch {start}', flush=True)
    if start >= epochs:
        raise ValueError('--epochs is the total target epoch count; checkpoint already reached it')
    config = {**config, 'max_steps': max_steps, 'val_steps': val_steps,
              'debug_run': bool(max_steps or val_steps), 'selection_metric': 'val_distance_mm',
              'validation_protocol': 'fixed_windows_v2', 'epochs': epochs, 'batch_size': batch_size,
              'lr': optimizer.param_groups[0]['lr'], 'resume': str(resume)}
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    try:
        for epoch in range(start, epochs):
            train.set_epoch(epoch)
            row = {'epoch': epoch}
            for split, dataset, cap in (('train', train, max_steps), ('val', val, val_steps)):
                model.train(split == 'train')
                generator = torch.Generator().manual_seed(config['seed'] + epoch)
                loader = DataLoader(dataset, batch_size=batch_size if split == 'train' else 1,
                                    shuffle=split == 'train', num_workers=workers, generator=generator)
                sums, count = {}, 0
                with torch.set_grad_enabled(split == 'train'):
                    for step, batch in enumerate(loader):
                        if split == 'train':
                            optimizer.zero_grad(set_to_none=True)
                        loss, metrics = loss_fn(model, batch)
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f'Non-finite {split} loss at epoch {epoch}, step {step}')
                        if split == 'train':
                            loss.backward()
                            optimizer.step()
                        n = len(batch['frames'])
                        for key, value in {'loss': loss, **metrics}.items():
                            scalar = float(value.detach())
                            if not np.isfinite(scalar):
                                raise FloatingPointError(f'Non-finite metric {key}')
                            sums[key] = sums.get(key, 0.) + scalar * n
                        count += n
                        if split == 'train' and (step + 1) % 50 == 0:
                            print(f'epoch={epoch} step={step+1}/{len(loader)} loss={float(loss.detach()):.5f}', flush=True)
                        if cap and step + 1 >= cap:
                            break
                if not count:
                    raise ValueError(f'{split} dataset is empty')
                row.update({f'{split}_{k}': v / count for k, v in sums.items()})
                row[f'{split}_windows'] = count
            improved = row['val_distance_mm'] < best
            best = min(best, row['val_distance_mm'])
            history.append(row)
            checkpoint = {'version': 2, 'method': config['method'], 'model': model.state_dict(),
                          'optimizer': optimizer.state_dict(), 'config': config, 'epoch': epoch,
                          'best': best, 'history': history, 'rng': rng_state()}
            atomic_save(checkpoint, output / 'last.pt')
            if improved:
                atomic_save(checkpoint, output / 'best.pt')
            temporary = output / 'history.json.tmp'
            temporary.write_text(json.dumps(history, indent=2) + '\n')
            temporary.replace(output / 'history.json')
            print(json.dumps(row), flush=True)
    finally:
        train.close()
        val.close()
