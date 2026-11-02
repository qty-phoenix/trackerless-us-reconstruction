"""Grouped overlapping video segments for case-wise DC2-Net supervision."""
import random
import torch
from torch.nn import functional as F
from datasets.longterm import LongTermDataset


class DC2Dataset(LongTermDataset):
    """Return K overlapping N-frame segments from one scan.

    Keeping the K predictions from the same scan together permits the paper's
    case-wise correlation loss instead of approximating it with unrelated
    scans in a shuffled batch.
    """

    def __init__(self, root, split='train', preprocessed=None, num_samples=5,
                 case_windows=4, size=(120, 160), seed=0, val_windows=3,
                 frame_cache=None):
        if case_windows < 2:
            raise ValueError('case_windows must be at least two for correlation')
        self.case_windows = case_windows
        span = num_samples + case_windows - 1
        super().__init__(root, split, preprocessed, num_samples, span, size,
                         seed, val_windows, frame_cache)

    def __getitem__(self, index):
        scan_index, window_index = divmod(index, self.val_windows)
        row = self.rows[scan_index]
        source = pose_source = None
        if self.frame_cache is None:
            source = self._file(self.root / row['frames_path'])
            pose_source = (source if row['tforms_path'] == row['frames_path'] else
                           self._file(self.root / row['tforms_path']))
            n = int(source['frames'].shape[0])
        else:
            n = int(row['num_frames'])
        span = self.sample_range
        if n < span:
            raise ValueError(f'{row["id"]} has only {n} frames')
        rng = random.Random(self.seed + scan_index * 1000003 +
                            (self.epoch * 1000000007 if self.split == 'train'
                             else window_index))
        if self.split == 'train':
            start = rng.randint(0, n - span)
        else:
            fraction = window_index / (self.val_windows - 1) if self.val_windows > 1 else .5
            start = round((n - span) * fraction)
        selected = torch.arange(start, start + span)
        if self.frame_cache is None:
            frames = torch.from_numpy(source['frames'][selected.numpy()].copy()).float().div_(255.)
            frames = F.interpolate(frames[:, None], size=(self.height, self.width),
                                   mode='bilinear', align_corners=False).squeeze(1)
            tforms = torch.from_numpy(pose_source['tforms'][selected.numpy()].copy()).float()
        else:
            frames = torch.from_numpy(self._cached_frames(row)[selected.numpy()].copy()).float().div_(255.)
            tforms = torch.from_numpy(self._cached_tforms(row)[selected.numpy()].copy())
        offsets = torch.arange(self.case_windows)[:, None] + torch.arange(self.num_samples)[None]
        return {'frames': frames[offsets], 'tforms': tforms[offsets],
                'indices': selected[offsets], 'scan_id': row['id'],
                'subject': row['subject']}
