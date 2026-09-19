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
    def __init__(self, root, split, preprocessed=None, num_samples=10, sample_range=10):
        self.root = Path(root).resolve()
        self.preprocessed = Path(preprocessed or self.root / 'preprocessed/longterm').resolve()
        self.rows = [json.loads(line) for line in (self.preprocessed / f'{split}.jsonl').read_text().splitlines() if line.strip()]
        self.num_samples, self.sample_range = num_samples, sample_range
        if num_samples < 2 or sample_range < num_samples:
            raise ValueError('Require sample_range >= num_samples >= 2')
        self.height, self.width = 480 // 4, 640 // 4
        self._files = {}

    def __len__(self):
        return len(self.rows)

    def _file(self, path):
        key = str(path)
        if key not in self._files:
            self._files[key] = h5py.File(path, 'r')
        return self._files[key]

    def __getitem__(self, index):
        row = self.rows[index]
        source = self._file(self.root / row['frames_path'])
        pose_source = source if row['tforms_path'] == row['frames_path'] else self._file(self.root / row['tforms_path'])
        n = int(source['frames'].shape[0])
        if n < self.sample_range:
            raise ValueError(f'{row["id"]} has only {n} frames')
        start = random.randint(0, n - self.sample_range)
        selected = sorted(random.sample(range(start, start + self.sample_range), self.num_samples))
        frames = torch.from_numpy(source['frames'][selected].copy()).float().div_(255.)
        # Match the author's frames_res4 spatial input while keeping temporal labels untouched.
        frames = F.interpolate(frames[:, None], size=(self.height, self.width), mode='bilinear', align_corners=False).squeeze(1)
        tforms = torch.from_numpy(pose_source['tforms'][selected].copy()).float()
        return {'frames': frames, 'tforms': tforms, 'indices': torch.tensor(selected),
                'scan_id': row['id'], 'subject': row['subject']}


def collate_longterm(batch):
    return {'frames': torch.stack([x['frames'] for x in batch]),
            'tforms': torch.stack([x['tforms'] for x in batch]),
            'indices': [x['indices'] for x in batch],
            'scan_id': [x['scan_id'] for x in batch], 'subject': [x['subject'] for x in batch]}
