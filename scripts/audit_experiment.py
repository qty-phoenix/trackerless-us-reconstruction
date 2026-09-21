#!/usr/bin/env python3
"""Verify completed experiment coverage and reconstruct metrics from raw logs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def close(actual, expected, name):
    if not np.allclose(actual, expected, rtol=1e-9, atol=1e-9):
        raise AssertionError(f'{name}: {actual} != {expected}')


def audit_eval(folder):
    summary = json.loads((folder / 'metrics.json').read_text())
    rows = [json.loads(x) for x in (folder / 'per_scan.jsonl').read_text().splitlines()]
    assert len(rows) == len({r['scan_id'] for r in rows}) == 72
    assert summary['scan_count'] == 72 and not summary['partial']
    for row in rows:
        scan = folder / row['scan_id']
        with np.load(scan / 'prediction.npz') as prediction, np.load(scan / 'reference.npz') as reference, np.load(scan / 'errors.npz') as errors:
            assert prediction['tool_global'].shape == reference['tool_to_world'].shape == (row['num_frames'], 4, 4)
            assert reference['landmarks'].shape == (20, 3)
            assert np.isfinite(prediction['tool_global']).all()
            close(prediction['tool_global'][0], np.eye(4), 'reference frame')
            for metric, raw in [('GPE', 'global_pixel_error_per_frame_mm'), ('LPE', 'local_pixel_error_per_frame_mm'),
                                ('GLE', 'global_landmark_errors_mm'), ('LLE', 'local_landmark_errors_mm')]:
                value = errors[raw]
                assert value.shape == ((row['num_frames'] - 1,) if 'pixel' in raw else (20,))
                assert np.isfinite(value).all() and (value >= 0).all()
                close(row[metric], value.mean(), f'{row["scan_id"]}/{metric}')
            for name in ('GL', 'LL'):
                value = np.linalg.norm(errors['pred_' + name] - errors['true_' + name], axis=0)
                close(value, errors['global_landmark_errors_mm' if name == 'GL' else 'local_landmark_errors_mm'], name)
            if 'field' in prediction:
                assert np.isfinite(prediction['field']).all()
                assert (prediction['bounds'][1] > prediction['bounds'][0]).all()
    for metric in ('GPE', 'GLE', 'LPE', 'LLE'):
        close(summary['mean'][metric], np.mean([r[metric] for r in rows]), metric)
        close(summary['std'][metric], np.std([r[metric] for r in rows]), metric + ' std')
    return {'scans': len(rows), 'frames': sum(r['num_frames'] for r in rows), 'all_raw_metrics_verified': True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('directory', type=Path)
    a = p.parse_args()
    root = a.directory
    status = json.loads((root / 'status.json').read_text())
    assert status['experiment']['state'] == 'complete'
    report = {}
    for method in ('longterm', 'nr_rec_fus'):
        report[method] = audit_eval(root / method / 'evaluation')
    report['nr_rec_fus_rigid'] = audit_eval(root / 'nr_rec_fus/evaluation_rigid')
    folder = root / 'nr_rec_fus/training'
    history = json.loads((folder / 'history.json').read_text())
    assert len(history) == 50
    totals, counts, batches = {}, {}, {}
    with (folder / 'steps.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if 'split' not in row:
                continue
            key = row['epoch'], row['split']
            assert len(row['scan_ids']) == len(row['frame_indices']) == row['windows']
            counts[key] = counts.get(key, 0) + row['windows']
            batches[key] = batches.get(key, 0) + 1
            total = totals.setdefault(key, {})
            for name, value in row['metrics'].items():
                assert np.isfinite(value)
                total[name] = total.get(name, 0.) + value * row['windows']
    for row in history:
        for split, windows, steps in (('train', 1200, 600), ('val', 216, 216)):
            key = row['epoch'], split
            assert counts[key] == windows and batches[key] == steps
            for name, value in totals[key].items():
                close(row[split + '_' + name], value / windows, f'{key}/{name}')
    best = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
    last = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
    assert last['epoch'] == 49 and not last['config']['debug_run']
    close(best['best'], min(row['val_distance_mm'] for row in history), 'best selection')
    report['training'] = {'epochs': 50, 'training_batches': 30000, 'validation_windows': 10800,
                          'best_epoch_zero_based': best['epoch'], 'all_epoch_means_verified': True}
    checksums = json.loads((root / 'artifacts_sha256.json').read_text())
    for name, expected in checksums.items():
        digest = hashlib.sha256()
        with (root / name).open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        assert digest.hexdigest() == expected, f'Checksum mismatch: {name}'
    report['artifact_hashes_verified'] = len(checksums)
    (root / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
