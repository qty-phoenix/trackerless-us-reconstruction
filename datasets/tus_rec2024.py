"""Lazy access to full-resolution scans/windows, without leaking tracked poses by default."""
import bisect
import json
from pathlib import Path

import h5py
import numpy as np


def load_calibration(path):
    lines = Path(path).read_text().splitlines()
    def matrix(name):
        start = lines.index(name) + 1
        return np.asarray([[float(v) for v in line.split(',')]
                           for line in lines[start:start + 4]], dtype=np.float64)
    return {'pixel_to_mm': matrix('scaling_from_pixel_to_mm'),
            'image_mm_to_tool': matrix('spatial_calibration_from_image_coordinate_system_to_tracking_tool_coordinate_system')}


class TUSREC2024:
    """NumPy dataset compatible with torch DataLoader.

    window=None returns full scans. Otherwise enumerate windows within each scan;
    a final overlapping window includes its last frame. Frames remain uint8 [N,H,W].
    include_targets=True returns tracker labels in a separate `targets` dictionary.
    Landmarks are returned for full scans, in the original official index convention.
    """
    def __init__(self, root, split='train', window=None, stride=1, include_targets=False):
        self.root = Path(root)
        if split not in ('train', 'val'):
            raise ValueError('Only public train/val splits are available; test is held by organizers')
        if window is not None and (window < 2 or stride < 1 or stride > window):
            raise ValueError('Require window >= 2 and 1 <= stride <= window')
        manifest = self.root / 'manifests' / f'{split}.jsonl'
        if not manifest.exists():
            raise FileNotFoundError(f'{manifest} is not ready; inspect data/tus-rec2024/status.json')
        self.rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        expected = 1200 if split == 'train' else 72
        if len(self.rows) != expected:
            raise ValueError(f'{split} is incomplete ({len(self.rows)}/{expected}); wait for preparation')
        self.window, self.include_targets = window, include_targets
        self.calibration = load_calibration(self.root / 'calib_matrix.csv')
        self.starts, self.cumulative = [], [0]
        for row in self.rows:
            n = row['num_frames']
            if window is None:
                starts = [0]
            elif n < window:
                starts = []
            else:
                starts = list(range(0, n - window + 1, stride))
                if starts[-1] != n - window:
                    starts.append(n - window)
            self.starts.append(starts)
            self.cumulative.append(self.cumulative[-1] + len(starts))

    def __len__(self):
        return self.cumulative[-1]

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        scan_index = bisect.bisect_right(self.cumulative, index) - 1
        row = self.rows[scan_index]
        start = self.starts[scan_index][index - self.cumulative[scan_index]]
        stop = start + self.window if self.window else row['num_frames']
        with h5py.File(self.root / row['frames_path'], 'r') as source:
            frames = source['frames'][start:stop]
        result = {'scan_id': row['id'], 'subject': row['subject'], 'frames': frames,
                  'frame_indices': np.arange(start, stop),
                  'calibration': {k: v.copy() for k, v in self.calibration.items()}}
        if self.window is None:
            with h5py.File(self.root / row['landmarks_path'], 'r') as source:
                result['landmarks'] = source[row['landmarks_key']][:]
        if self.include_targets:
            with h5py.File(self.root / row['tforms_path'], 'r') as source:
                result['targets'] = {'tool_to_world': source['tforms'][start:stop].astype(np.float64)}
        return result
