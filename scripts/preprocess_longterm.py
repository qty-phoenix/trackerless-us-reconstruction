#!/usr/bin/env python3
"""Create the reproducible, split-safe input index for the Long-Term Dependency baseline.

This does not duplicate the raw HDF5 frames. It records the official scan paths,
the 4x target resolution used by the paper implementation, calibration in the
downsampled pixel coordinate system, and subject-level train/validation splits.
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np


def read_calibration(path, factor):
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line and ',' in line and line[0].isdigit() or line.startswith('-'):
            rows.append([float(x) for x in line.split(',')])
    if len(rows) != 8 or any(len(row) != 4 for row in rows):
        raise ValueError(f'Expected 8x4 calibration matrix in {path}')
    s, c = np.asarray(rows[:4], np.float32), np.asarray(rows[4:], np.float32)
    # Raw pixel -> image mm -> tool; raw pixel coordinates = factor * reduced coordinates.
    resize = np.diag([factor, factor, 1, 1]).astype(np.float32)
    return c @ s @ resize


def build(root, output, factor=4, seed=0):
    root = Path(root).resolve()
    output = Path(output).resolve()
    manifest_train = root / 'manifests/train.jsonl'
    manifest_val = root / 'manifests/val.jsonl'
    if not manifest_train.exists() or not manifest_val.exists():
        raise FileNotFoundError('Run the dataset preparation first; train.jsonl and val.jsonl are required')
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        'method': 'Long-Term Dependency for 3D Reconstruction of Freehand Ultrasound Without External Tracker',
        'paper_input_frames': 10,
        'paper_sample_range': 10,
        'paper_num_pred': 9,
        'paper_pred_pairs': 45,
        'resample_factor': factor,
        'raw_frame_size_hw': [480, 640],
        'model_frame_size_hw': [480 // factor, 640 // factor],
        'split_policy': 'official TUS-REC2024 subject split: train 000-049, val 050-052',
        'seed': seed,
        'labels': 'tool-frame point coordinates; tracker poses are never returned by inference dataset',
    }
    calibration = read_calibration(root / 'calib_matrix.csv', factor)
    np.save(output / 'image_to_tool_downsampled.npy', calibration)
    for split, source in [('train', manifest_train), ('val', manifest_val)]:
        rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        rows = [{**row, 'root_relative_frames_path': row['frames_path'],
                 'model_height': 480 // factor, 'model_width': 640 // factor,
                 'resample_factor': factor, 'num_samples': 10, 'sample_range': 10,
                 'num_pred': 9} for row in rows]
        (output / f'{split}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        metadata[f'{split}_scans'] = len(rows)
        metadata[f'{split}_subjects'] = sorted({row['subject'] for row in rows})
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'calibration': str(output / 'image_to_tool_downsampled.npy'),
                      'train_scans': metadata['train_scans'], 'val_scans': metadata['val_scans']}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('data/tus-rec2024'))
    p.add_argument('--output', type=Path, default=Path('data/tus-rec2024/preprocessed/longterm'))
    p.add_argument('--factor', type=int, default=4)
    args = p.parse_args()
    build(args.root, args.output, args.factor)
