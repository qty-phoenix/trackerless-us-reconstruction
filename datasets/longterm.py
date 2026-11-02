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
                 size=(120, 160), seed=0, val_windows=3, frame_cache=None):
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
        self.frame_cache = Path(frame_cache).resolve() if frame_cache else None
        self._frame_files, self._tform_files = {}, {}
        if self.frame_cache is not None:
            metadata = json.loads((self.frame_cache / 'metadata.json').read_text())
            if (metadata.get('version', 0) < 2 or not metadata.get('complete') or
                    metadata.get('height') != self.height or metadata.get('width') != self.width or
                    metadata.get('dtype') != 'uint8' or metadata.get('tforms_dtype') != 'float32' or
                    Path(metadata.get('root', '')).resolve() != self.root):
                raise ValueError(f'Incomplete or incompatible frame cache: {self.frame_cache}')

    def __len__(self):
        return len(self.rows) * self.val_windows

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _file(self, path):
        key = str(path)
        if key not in self._files:
            self._files[key] = h5py.File(path, 'r')
        return self._files[key]

    def _cached_frames(self, row):
        key = row['id']
        if key not in self._frame_files:
            path = self.frame_cache / 'frames' / f'{key}.npy'
            value = np.load(path, mmap_mode='r')
            if value.shape != (int(row['num_frames']), self.height, self.width) or value.dtype != np.uint8:
                raise ValueError(f'Invalid cached frames: {path}')
            self._frame_files[key] = value
        return self._frame_files[key]

    def _cached_tforms(self, row):
        key = row['id']
        if key not in self._tform_files:
            path = self.frame_cache / 'tforms' / f'{key}.npy'
            value = np.load(path, mmap_mode='r')
            if value.shape != (int(row['num_frames']), 4, 4) or value.dtype != np.float32:
                raise ValueError(f'Invalid cached transforms: {path}')
            self._tform_files[key] = value
        return self._tform_files[key]

    def __getitem__(self, index):
        scan_index, window_index = divmod(index, self.val_windows)
        row = self.rows[scan_index]
        source = pose_source = None
        if self.frame_cache is None:
            source = self._file(self.root / row['frames_path'])
            pose_source = source if row['tforms_path'] == row['frames_path'] else self._file(self.root / row['tforms_path'])
            n = int(source['frames'].shape[0])
        else:
            n = int(row['num_frames'])
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
        if self.frame_cache is None:
            frames = torch.from_numpy(source['frames'][selected].copy()).float().div_(255.)
            # Match the author's frames_res4 spatial input while keeping temporal labels untouched.
            frames = F.interpolate(frames[:, None], size=(self.height, self.width), mode='bilinear', align_corners=False).squeeze(1)
            tforms = torch.from_numpy(pose_source['tforms'][selected].copy()).float()
        else:
            frames = torch.from_numpy(self._cached_frames(row)[selected].copy()).float().div_(255.)
            tforms = torch.from_numpy(self._cached_tforms(row)[selected].copy())
        return {'frames': frames, 'tforms': tforms, 'indices': torch.tensor(selected),
                'scan_id': row['id'], 'subject': row['subject']}

    def close(self):
        for source in self._files.values():
            source.close()
        self._files.clear()
        self._frame_files.clear()
        self._tform_files.clear()

    def __getstate__(self):
        return {**self.__dict__, '_files': {}, '_frame_files': {}, '_tform_files': {}}


def collate_longterm(batch):
    return {'frames': torch.stack([x['frames'] for x in batch]),
            'tforms': torch.stack([x['tforms'] for x in batch]),
            'indices': [x['indices'] for x in batch],
            'scan_id': [x['scan_id'] for x in batch], 'subject': [x['subject'] for x in batch]}
