#!/usr/bin/env python3
"""Validate and finalize a MoGLo experiment after training/evaluation completed."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def close(actual, expected, label):
    if not np.allclose(actual, expected, rtol=1e-8, atol=1e-8):
        raise AssertionError(f'{label}: {actual} != {expected}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory.resolve()
    method = root / 'moglo_net'
    training = method / 'training'
    evaluation = method / 'evaluation'
    history = json.loads((training / 'history.json').read_text())
    if len(history) != 100 or history[-1]['epoch'] != 99:
        raise AssertionError('Expected all 100 epochs')
    rows = [json.loads(line) for line in (evaluation / 'per_scan.jsonl').read_text().splitlines()]
    metrics = json.loads((evaluation / 'metrics.json').read_text())
    if len(rows) != 72 or len({row['scan_id'] for row in rows}) != 72 or metrics['partial']:
        raise AssertionError('Expected 72 unique complete validation scans')
    frames = 0
    for row in rows:
        frames += row['num_frames']
        scan = evaluation / row['scan_id']
        with np.load(scan / 'prediction.npz') as prediction, np.load(scan / 'reference.npz') as reference, np.load(scan / 'errors.npz') as errors:
            if prediction['tool_global'].shape != reference['tool_to_world'].shape:
                raise AssertionError(f'Pose shape mismatch: {row["scan_id"]}')
            if reference['landmarks'].shape != (20, 3):
                raise AssertionError(f'Landmark shape mismatch: {row["scan_id"]}')
            for name, raw in (('GPE', 'global_pixel_error_per_frame_mm'),
                              ('LPE', 'local_pixel_error_per_frame_mm'),
                              ('GLE', 'global_landmark_errors_mm'),
                              ('LLE', 'local_landmark_errors_mm')):
                close(row[name], errors[raw].mean(), f'{row["scan_id"]}/{name}')
    for name in ('GPE', 'GLE', 'LPE', 'LLE'):
        close(metrics['mean'][name], np.mean([row[name] for row in rows]), name)
    step_counts = {(epoch, split): 0 for epoch in range(100) for split in ('train', 'val')}
    with (training / 'steps.jsonl').open() as stream:
        for line in stream:
            record = json.loads(line)
            if 'split' in record:
                step_counts[record['epoch'], record['split']] += record['windows']
    for epoch in range(100):
        if step_counts[epoch, 'train'] != 1200 or step_counts[epoch, 'val'] != 216:
            raise AssertionError(f'Incomplete raw steps at epoch {epoch + 1}')
    best = torch.load(training / 'best.pt', map_location='cpu', weights_only=False)
    last = torch.load(training / 'last.pt', map_location='cpu', weights_only=False)
    best_value, best_epoch = min((row['val_distance_mm'], row['epoch']) for row in history)
    if best['epoch'] != best_epoch or last['epoch'] != 99:
        raise AssertionError('Checkpoint epoch mismatch')
    close(best['best'], best_value, 'best checkpoint selection')
    with (root / 'per_scan_comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(root / 'comparison.json', {'moglo_net': metrics})
    report = ['# MoGLo-Net TUS-REC2024 experiment', '',
              '| Method | GPE (mm) | GLE (mm) | LPE (mm) | LLE (mm) |',
              '| --- | ---: | ---: | ---: | ---: |',
              '| moglo_net | ' + ' | '.join(f"{metrics['mean'][name]:.6f}" for name in ('GPE', 'GLE', 'LPE', 'LLE')) + ' |', '',
              f'Training completed 100 epochs; the selected checkpoint is epoch {best_epoch + 1} with fixed-window validation distance {best_value:.6f} mm.',
              f'All 72 validation scans ({frames} frames) and every original-resolution pixel were evaluated.',
              'This is a TUS-REC2024 adaptation of the public MoGLo-Net implementation, not a numerical reproduction of the paper Forearm_Main experiment.',
              'Raw records include environment, source hashes, command logs, steps.jsonl, history.json, loss_curves.png/svg, and per-scan prediction/reference/error files.']
    (root / 'REPORT.md').write_text('\n'.join(report) + '\n')
    verification = {'epochs': 100, 'best_epoch_one_based': best_epoch + 1,
                    'best_val_distance_mm': best_value, 'training_windows': 120000,
                    'validation_windows': 21600, 'scans': 72, 'frames': frames,
                    'raw_metrics_verified': True, 'raw_steps_verified': True}
    status = json.loads((root / 'status.json').read_text())
    status['experiment'].update(state='complete', error=None)
    status['moglo_net'].update(state='complete', phase='complete', error=None,
                               metrics=metrics['mean'], checkpoint=str(training / 'best.pt'),
                               checkpoint_sha256=digest(training / 'best.pt'))
    status['moglo_net'].pop('traceback', None)
    write_json(root / 'status.json', status)
    included = [path for path in sorted(root.rglob('*'))
                if path.is_file() and path.name not in ('artifacts_sha256.json', 'status.json', 'gpu_telemetry.jsonl')]
    # verification.json itself is included after it is written.
    verification['artifact_hashes'] = len(included) + (0 if (root / 'verification.json').exists() else 1)
    write_json(root / 'verification.json', verification)
    checksums = {}
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.name not in ('artifacts_sha256.json', 'status.json', 'gpu_telemetry.jsonl'):
            checksums[str(path.relative_to(root))] = digest(path)
    write_json(root / 'artifacts_sha256.json', checksums)
    print(json.dumps({'metrics': metrics['mean'], **verification}, indent=2))


if __name__ == '__main__':
    main()
