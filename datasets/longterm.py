"""Lazy TUS-REC2024 loader matching the Long-Term Dependency training protocol."""
import json
from pathlib import Path
import random
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class LongTermDataset(Dataset):
    def __init__(self, root, split, preprocessed=None, num_samples=10, sample_range=10,
                 size=(120, 160), seed=0, val_windows=3):
        self.root = Path(root).resolve()
        self.preprocessed = Path(preprocessed or self.root / 'preprocessed/longterm').resolve()
        self.rows = [json.loads(line) for line in (self.preprocessed / f'{split}.jsonl').read_text().splitlines() if line.strip()]
        self.num_samples, self.sample_range = num_samples, sample_range
        if num_samples < 2 or sample_range < num_samples:
            raise ValueError('Require sample_range >= num_samples >= 2')
        if split not in ('train', 'val') or val_windows < 1:
            raise ValueError('Require train/val split and val_windows >= 1')
        self.height, self.width = size
        self.split, self.seed, self.epoch = split, seed, 0
        self.val_windows = val_windows if split == 'val' else 1
        self._files = {}

    def __len__(self):
        return len(self.rows) * self.val_windows

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _file(self, path):
        key = str(path)
        if key not in self._files:
            self._files[key] = h5py.File(path, 'r')
        return self._files[key]

    def __getitem__(self, index):
        scan_index, window_index = divmod(index, self.val_windows)
        row = self.rows[scan_index]
        source = self._file(self.root / row['frames_path'])
        pose_source = source if row['tforms_path'] == row['frames_path'] else self._file(self.root / row['tforms_path'])
        n = int(source['frames'].shape[0])
        if n < self.sample_range:
            raise ValueError(f'{row["id"]} has only {n} frames')
        rng = random.Random(self.seed + scan_index * 1000003 +
                            (self.epoch * 1000000007 if self.split == 'train' else window_index))
        if self.split == 'train':
            start = rng.randint(0, n - self.sample_range)
        else:
            start = round((n - self.sample_range) *
                          (window_index / (self.val_windows - 1) if self.val_windows > 1 else .5))
        selected = sorted(rng.sample(range(start, start + self.sample_range), self.num_samples))
        frames = torch.from_numpy(source['frames'][selected].copy()).float().div_(255.)
        # Match the author's frames_res4 spatial input while keeping temporal labels untouched.
        frames = F.interpolate(frames[:, None], size=(self.height, self.width), mode='bilinear', align_corners=False).squeeze(1)
        tforms = torch.from_numpy(pose_source['tforms'][selected].copy()).float()
        return {'frames': frames, 'tforms': tforms, 'indices': torch.tensor(selected),
                'scan_id': row['id'], 'subject': row['subject']}

    def close(self):
        for source in self._files.values():
            source.close()
        self._files.clear()

    def __getstate__(self):
        return {**self.__dict__, '_files': {}}


def collate_longterm(batch):
    return {'frames': torch.stack([x['frames'] for x in batch]),
            'tforms': torch.stack([x['tforms'] for x in batch]),
            'indices': [x['indices'] for x in batch],
            'scan_id': [x['scan_id'] for x in batch], 'subject': [x['subject'] for x in batch]}
