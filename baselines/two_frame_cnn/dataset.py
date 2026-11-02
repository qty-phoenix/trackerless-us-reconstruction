"""Deterministic scan-wise pair sampling for the 2-Frame CNN."""
import random
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from datasets.longterm import LongTermDataset


class TwoFrameDataset(LongTermDataset):
    """Return one augmented pair per training scan and fixed validation pairs.

    This is the ``epoch_mode=tus_rec`` protocol published with DualTrack's
    reproduction: one pair per sweep per epoch, optional one-frame skipping,
    order reversal and horizontal reflection.  Validation has no augmentation.
    """

    def __init__(self, root, split='train', preprocessed=None, size=(128, 128),
                 seed=0, val_windows=3, flip_h_prob=0.5,
                 reverse_sweep_prob=0.5, skip_frame_prob=0.2, frame_cache=None,
                 pairs_per_scan=1):
        if tuple(size) not in ((128, 128), (120, 160)):
            raise ValueError('TwoFrameCNN supports 128x128 or FairBench 120x160 inputs')
        for name, value in (('flip_h_prob', flip_h_prob),
                            ('reverse_sweep_prob', reverse_sweep_prob),
                            ('skip_frame_prob', skip_frame_prob)):
            if not 0 <= value <= 1:
                raise ValueError(f'{name} must be in [0,1]')
        if pairs_per_scan < 1:
            raise ValueError('pairs_per_scan must be positive')
        super().__init__(root, split, preprocessed, 2, 2, size, seed, val_windows)
        self.flip_h_prob = flip_h_prob
        self.reverse_sweep_prob = reverse_sweep_prob
        self.skip_frame_prob = skip_frame_prob
        self.pairs_per_scan = pairs_per_scan if split == 'train' else 1
        self._scan_orders = []
        self.frame_cache = Path(frame_cache).resolve() if frame_cache else None
        self._frame_files = {}
        self._tform_files = {}
        if self.frame_cache is not None:
            metadata_path = self.frame_cache / 'metadata.json'
            if not metadata_path.exists():
                raise FileNotFoundError(f'Run scripts/preprocess_two_frame_cnn.py: {metadata_path}')
            metadata = json.loads(metadata_path.read_text())
            if (metadata.get('version', 0) < 2 or not metadata.get('complete') or
                    metadata.get('height') != self.height or
                    metadata.get('width') != self.width or metadata.get('dtype') != 'uint8' or
                    metadata.get('tforms_dtype') != 'float32' or
                    Path(metadata.get('root', '')).resolve() != self.root):
                raise ValueError(f'Incomplete or incompatible frame cache: {metadata_path}')
        self.set_epoch(0)

    def __len__(self):
        if self.split == 'train':
            return len(self.rows) * self.pairs_per_scan
        return super().__len__()

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        if self.split == 'train':
            # Each packed slot exactly reproduces one original scan-wise epoch:
            # its pair RNG and scan permutation use that virtual epoch number.
            self._scan_orders = []
            for slot in range(self.pairs_per_scan):
                virtual_epoch = epoch * self.pairs_per_scan + slot
                generator = torch.Generator().manual_seed(self.seed + virtual_epoch)
                # DataLoader consumes one int64 from its generator for worker
                # base seeding before RandomSampler draws this permutation.
                torch.empty((), dtype=torch.int64).random_(generator=generator)
                self._scan_orders.append(torch.randperm(len(self.rows),
                                                        generator=generator).tolist())

    def _cached(self, row):
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
        if self.split == 'train':
            slot, position = divmod(index, len(self.rows))
            scan_index = self._scan_orders[slot][position]
            virtual_epoch = self.epoch * self.pairs_per_scan + slot
            window_index = 0
        else:
            scan_index, window_index = divmod(index, self.val_windows)
            virtual_epoch = 0
        row = self.rows[scan_index]
        source = pose_source = None
        if self.frame_cache is None:
            source = self._file(self.root / row['frames_path'])
            pose_source = (source if row['tforms_path'] == row['frames_path'] else
                           self._file(self.root / row['tforms_path']))
            n = int(source['frames'].shape[0])
        else:
            n = int(row['num_frames'])
        if n < 2:
            raise ValueError(f'{row["id"]} has only {n} frames')
        rng = random.Random(self.seed + scan_index * 1000003 +
                            (virtual_epoch * 1000000007 if self.split == 'train'
                             else window_index))
        if self.split == 'train':
            offset = 2 if n >= 3 and rng.random() < self.skip_frame_prob else 1
            start = rng.randint(0, n - offset - 1)
            reverse = rng.random() < self.reverse_sweep_prob
            flipped = rng.random() < self.flip_h_prob
        else:
            offset, reverse, flipped = 1, False, False
            fraction = window_index / (self.val_windows - 1) if self.val_windows > 1 else .5
            start = round((n - 2) * fraction)
        selected = torch.tensor([start, start + offset])
        if self.frame_cache is None:
            frames = torch.from_numpy(source['frames'][selected.numpy()].copy()).float().div_(255.)
            frames = F.interpolate(frames[:, None], size=(self.height, self.width),
                                   mode='bilinear', align_corners=False).squeeze(1)
        else:
            frames = torch.from_numpy(self._cached(row)[selected.numpy()].copy()).float().div_(255.)
        if self.frame_cache is None:
            tforms = torch.from_numpy(pose_source['tforms'][selected.numpy()].copy()).float()
        else:
            tforms = torch.from_numpy(self._cached_tforms(row)[selected.numpy()].copy())
        if reverse:
            frames, tforms, selected = frames.flip(0), tforms.flip(0), selected.flip(0)
        if flipped:
            frames = frames.flip(-1)
        return {'frames': frames, 'tforms': tforms, 'indices': selected,
                'flipped': torch.tensor(flipped), 'scan_id': row['id'],
                'subject': row['subject']}

    def close(self):
        super().close()
        self._frame_files.clear()
        self._tform_files.clear()

    def __getstate__(self):
        return {**super().__getstate__(), '_frame_files': {}, '_tform_files': {}}
