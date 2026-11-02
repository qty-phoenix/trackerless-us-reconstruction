#!/usr/bin/env python3
"""Full-resolution, full-scan geometric evaluation; mean each metric over scans."""
import argparse
import json
import time
from pathlib import Path
import h5py
import numpy as np
import torch
from geometry import ScanPrediction, score_scan
from inference import Predictor, export_ddfs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--method', choices=('longterm', 'nr_rec_fus', 'moglo_net', 'recon',
                                        'dc2_net', 'two_frame_cnn'))
    p.add_argument('--root', default='data/tus-rec2024')
    p.add_argument('--split', choices=('train', 'val'), default='val')
    p.add_argument('--manifest', default='',
                   help='Explicit JSONL manifest; overrides --split and permits benchmark holdouts')
    p.add_argument('--device', default='cuda')
    p.add_argument('--metric-device', default=None, help='Defaults to --device; use cpu for CPU scoring')
    p.add_argument('--output', required=True)
    p.add_argument('--max-scans', type=int, default=0, help='Debug subset; 0 evaluates the entire split')
    p.add_argument('--chunk-size', type=int, default=16384)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--rigid-only', action='store_true', help='NR rigid-branch ablation')
    p.add_argument('--online-epochs', type=int, default=None,
                   help='RecON debug/ablation override; formal protocol uses checkpoint value 30')
    p.add_argument('--export-ddfs', action='store_true', help='Write full-resolution GP/GL/LP/LL per scan')
    a = p.parse_args()
    if a.max_scans < 0 or a.chunk_size < 1 or a.cpu_threads < 1:
        p.error('Invalid evaluation limits')
    torch.set_num_threads(a.cpu_threads)
    root, output = Path(a.root), Path(a.output)
    if (output / 'metrics.json').exists() or (output / 'per_scan.jsonl').exists():
        raise FileExistsError(f'{output} already has evaluation results; use a new --output')
    manifest = Path(a.manifest) if a.manifest else root / 'manifests' / f'{a.split}.jsonl'
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    expected = len(rows) if a.manifest else (72 if a.split == 'val' else 1200)
    if not rows or len(rows) != expected:
        raise ValueError(f'Incomplete manifest {manifest}: {len(rows)}/{expected}')
    selected = rows[:a.max_scans] if a.max_scans else rows
    predictor = Predictor(a.checkpoint, a.device, a.method, a.rigid_only, a.online_epochs)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for row in selected:
        started = time.perf_counter()
        # The predictor receives only frames and calibration. Read labels later.
        with h5py.File(root / row['frames_path'], 'r') as source:
            prediction = predictor.predict_scan(source['frames'], root / 'calib_matrix.csv')
        inference_seconds = time.perf_counter() - started
        with h5py.File(root / row['tforms_path'], 'r') as source:
            tforms = source['tforms'][:]
        with h5py.File(root / row['landmarks_path'], 'r') as source:
            landmarks = source[row['landmarks_key']][:]
        scoring_started = time.perf_counter()
        metric_device = a.metric_device or a.device
        scoring_prediction = ScanPrediction(prediction.tool_global.to(metric_device),
                                            prediction.pixel_to_mm.to(metric_device),
                                            prediction.image_mm_to_tool.to(metric_device),
                                            prediction.field.to(metric_device) if prediction.field is not None else None,
                                            tuple(x.to(metric_device) for x in prediction.bounds) if prediction.bounds else None)
        metrics, details = score_scan(scoring_prediction, tforms, landmarks, chunk_size=a.chunk_size, return_details=True)
        result = {'scan_id': row['id'], 'num_frames': len(tforms), **metrics,
                  'inference_seconds': inference_seconds,
                  'scoring_seconds': time.perf_counter() - scoring_started}
        results.append(result)
        print(json.dumps(result), flush=True)
        with (output / 'per_scan.jsonl').open('a') as stream:
            stream.write(json.dumps(result) + '\n')
        scan_dir = output / row['id']
        scan_dir.mkdir(parents=True, exist_ok=True)
        compact = {'tool_global': prediction.tool_global.numpy(),
                   'pixel_to_mm': prediction.pixel_to_mm.numpy(),
                   'image_mm_to_tool': prediction.image_mm_to_tool.numpy()}
        if prediction.field is not None:
            compact.update(field=prediction.field.numpy(), bounds=np.stack([x.numpy() for x in prediction.bounds]))
        np.savez_compressed(scan_dir / 'prediction.npz', **compact)
        np.savez_compressed(scan_dir / 'reference.npz', tool_to_world=tforms, landmarks=landmarks)
        np.savez_compressed(scan_dir / 'errors.npz', **details)
        if a.export_ddfs:
            export_ddfs(prediction, landmarks, scan_dir)
    summary = {'method': predictor.method, 'checkpoint': str(Path(a.checkpoint).resolve()),
               'checkpoint_config': predictor.config,
               'split': manifest.stem if a.manifest else a.split,
               'manifest': str(manifest.resolve()), 'scan_count': len(results),
               'partial': len(results) != len(rows), 'rigid_only': a.rigid_only,
               'metric_device': metric_device, 'chunk_size': a.chunk_size,
               'units': 'mm', 'aggregation': 'unweighted mean over scans',
               'pixel_coordinates': 'official x=1..640, y=1..480, x-fastest',
               'stitching': ('full-sequence adjacent transforms with per-scan online adaptation'
                             if predictor.method == 'recon' else
                             'all overlapping windows; average mean-motion estimates on covered edges'
                             if predictor.method == 'dc2_net' else
                             'all adjacent frame pairs; compose predicted image transforms'
                             if predictor.method == 'two_frame_cnn' else
                             'stride K-1; anchor window first; retain overlap; include tail'),
               'nr_local_definition': 'difference of anchored refined global positions in previous rigid image basis',
               'mean': {k: float(np.mean([r[k] for r in results])) for k in ('GPE', 'GLE', 'LPE', 'LLE', 'inference_seconds')},
               'std': {k: float(np.std([r[k] for r in results])) for k in ('GPE', 'GLE', 'LPE', 'LLE')}}
    (output / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary['mean']), flush=True)


if __name__ == '__main__':
    main()
