"""Official-shaped entry point. Set TUSREC_CHECKPOINT before importing/calling."""
from functools import lru_cache
import os
from inference import Predictor


@lru_cache(maxsize=1)
def _predictor(checkpoint, device):
    return Predictor(checkpoint, device)


def predict_ddfs(frames, landmark, data_path_calib):
    checkpoint = os.environ.get('TUSREC_CHECKPOINT')
    if not checkpoint:
        raise ValueError('Set TUSREC_CHECKPOINT to a trained best.pt checkpoint')
    return _predictor(checkpoint, os.environ.get('TUSREC_DEVICE', 'cuda')).predict_ddfs(
        frames, landmark, data_path_calib)
