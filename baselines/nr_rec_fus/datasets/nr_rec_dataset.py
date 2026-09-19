"""Share split-safe HDF5 loading and deterministic validation with Long-Term."""
from datasets.longterm import LongTermDataset


class NRRecDataset(LongTermDataset):
    def __init__(self, root, split='train', preprocessed=None, num_samples=4,
                 sample_range=4, size=(120, 160), seed=0, val_windows=3):
        super().__init__(root, split, preprocessed, num_samples, sample_range,
                         size, seed, val_windows)
