"""Split-safe RecON windows on TUS-REC2024."""
from datasets.longterm import LongTermDataset
from .models import pair_inputs


class RecONDataset(LongTermDataset):
    """Variable-motion windows sampled only inside one subject/scan."""
    def __init__(self, *args, with_inputs=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.with_inputs = with_inputs

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if self.with_inputs:
            # Keep CPU OpenCV work inside DataLoader workers. The resulting
            # values are identical to main-process preprocessing, but overlap
            # optical-flow computation with GPU training.
            sample['inputs'] = pair_inputs(sample['frames'])
        return sample
