#!/usr/bin/env python3
"""Build resumable uncompressed resized-frame caches for the 2-Frame CNN."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training import select_device


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def valid_cache(path, shape, dtype):
    if not path.exists():
        return False
    value = np.load(path, mmap_mode='r')
    valid = value.shape == shape and value.dtype == np.dtype(dtype)
    del value
    if not valid:
        raise ValueError(f'Existing cache has the wrong shape or dtype: {path}')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data/tus-rec2024')
    parser.add_argument('--preprocessed', default='data/tus-rec2024/preprocessed/longterm')
    parser.add_argument('--output', default='data/tus-rec2024/preprocessed/two_frame_cnn')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--height', type=int, default=128)
    parser.add_argument('--width', type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1 or (args.height, args.width) not in ((128, 128), (120, 160)):
        parser.error('Require a positive batch and either 128x128 or 120x160 output')
    root, index_root, output = map(Path, (args.root, args.preprocessed, args.output))
    device = select_device(args.device)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {'version': 2, 'complete': False, 'height': args.height, 'width': args.width,
                'dtype': 'uint8', 'tforms_dtype': 'float32',
                'interpolation': 'pytorch_bilinear_align_corners_false_round',
                'root': str(root.resolve()), 'device': str(device), 'batch_size': args.batch_size,
                'started_utc': datetime.now(timezone.utc).isoformat(), 'files': {}}
    metadata_path = output / 'metadata.json'
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text())
        if old.get('complete') and old.get('version', 0) >= 2:
            print(json.dumps({'output': str(output), 'already_complete': True}), flush=True)
            return
        metadata['started_utc'] = old.get('started_utc', metadata['started_utc'])
        metadata['files'].update(old.get('files', {}))
    write_json(metadata_path, metadata)
    totals = {}
    try:
        for split in ('train', 'val'):
            rows = [json.loads(line) for line in (index_root / f'{split}.jsonl').read_text().splitlines()
                    if line.strip()]
            frame_count = 0
            for number, row in enumerate(rows, 1):
                source_path = root / row['frames_path']
                target = output / 'frames' / f"{row['id']}.npy"
                target.parent.mkdir(parents=True, exist_ok=True)
                shape = (int(row['num_frames']), args.height, args.width)
                if not valid_cache(target, shape, np.uint8):
                    partial = target.with_suffix('.npy.partial')
                    cache = np.lib.format.open_memmap(partial, mode='w+', dtype=np.uint8,
                                                      shape=shape)
                    with h5py.File(source_path, 'r') as source:
                        if tuple(source['frames'].shape) != (shape[0], 480, 640):
                            raise ValueError(f'Unexpected frame shape in {source_path}')
                        for start in range(0, shape[0], args.batch_size):
                            stop = min(start + args.batch_size, shape[0])
                            frames = torch.as_tensor(source['frames'][start:stop], device=device)
                            frames = F.interpolate(frames[:, None].float(), size=(args.height, args.width),
                                                   mode='bilinear', align_corners=False)[:, 0]
                            cache[start:stop] = frames.round().clamp_(0, 255).byte().cpu().numpy()
                    cache.flush()
                    del cache
                    partial.replace(target)
                tforms_target = output / 'tforms' / f"{row['id']}.npy"
                tforms_target.parent.mkdir(parents=True, exist_ok=True)
                tforms_shape = (shape[0], 4, 4)
                if not valid_cache(tforms_target, tforms_shape, np.float32):
                    with h5py.File(root / row['tforms_path'], 'r') as source:
                        tforms = source['tforms'][:]
                    if tforms.shape != tforms_shape:
                        raise ValueError(f'Unexpected transform shape in {row["tforms_path"]}')
                    partial = tforms_target.with_suffix('.npy.partial')
                    with partial.open('wb') as stream:
                        np.save(stream, tforms.astype(np.float32, copy=False))
                    partial.replace(tforms_target)
                metadata['files'][row['id']] = {'split': split, 'frames': shape[0],
                                                 'path': str(target.relative_to(output)),
                                                 'bytes': target.stat().st_size,
                                                 'tforms_path': str(tforms_target.relative_to(output)),
                                                 'tforms_bytes': tforms_target.stat().st_size}
                frame_count += shape[0]
                metadata.update(last_scan=row['id'], processed_files=len(metadata['files']),
                                updated_utc=datetime.now(timezone.utc).isoformat())
                write_json(metadata_path, metadata)
                if number % 25 == 0 or number == len(rows):
                    print(f'{split}: {number}/{len(rows)} scans', flush=True)
            totals[split] = {'scans': len(rows), 'frames': frame_count}
        metadata.update(complete=True, totals=totals,
                        completed_utc=datetime.now(timezone.utc).isoformat())
        write_json(metadata_path, metadata)
        print(json.dumps({'output': str(output), 'totals': totals,
                          'bytes': sum(item['bytes'] + item['tforms_bytes']
                                       for item in metadata['files'].values())}),
              flush=True)
    except BaseException:
        metadata.update(complete=False, failed_utc=datetime.now(timezone.utc).isoformat())
        write_json(metadata_path, metadata)
        raise


if __name__ == '__main__':
    main()
