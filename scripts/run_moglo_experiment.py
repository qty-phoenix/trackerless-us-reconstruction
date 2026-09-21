#!/usr/bin/env python3
"""Train and fully evaluate the MoGLo-Net TUS-REC2024 adaptation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import os
import shutil
import sys
import threading
import time
import traceback
import subprocess

from run_baseline_experiment import (REPO, command, evaluate, sha256, snapshot,
                                     telemetry, update, utc, write_json)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--config', default='baselines/moglo_net/configs/tus_rec2024.json')
    args = p.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    method = 'moglo_net'
    folder = out / method
    folder.mkdir()
    update(out, 'experiment', state='initializing')
    done = threading.Event()
    monitor = None
    try:
        snapshot(out)
        config = json.loads((REPO / args.config).read_text())
        config['output'] = str(folder / 'training')
        config_path = folder / 'train_config.json'
        write_json(config_path, config)
        monitor = threading.Thread(target=telemetry, args=(out, done), daemon=True)
        monitor.start()
        command(out, method, args.gpu, 'training', ['-m', 'baselines.moglo_net.train',
                '--config', str(config_path), '--device', 'cuda', '--workers', '0'])
        checkpoint = folder / 'training/best.pt'
        history = json.loads((folder / 'training/history.json').read_text())
        if len(history) != config['epochs'] or any(r['train_windows'] != 1200 or r['val_windows'] != 216 for r in history):
            raise RuntimeError('MoGLo-Net training history does not cover all windows')
        metrics = evaluate(out, method, args.gpu, checkpoint)
        status = {'state': 'complete', 'phase': 'complete', 'gpu': args.gpu,
                  'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint),
                  'metrics': metrics['mean'], 'updated_utc': utc()}
        update(out, method, **status)
        update(out, 'experiment', state='complete', ended_utc=utc())
        lines = ['# MoGLo-Net TUS-REC2024 experiment', '', f'Completed UTC: {utc()}', '',
                 '| Method | GPE (mm) | GLE (mm) | LPE (mm) | LLE (mm) |', '| --- | ---: | ---: | ---: | ---: |',
                 '| moglo_net | ' + ' | '.join(f"{metrics['mean'][key]:.6f}" for key in ('GPE', 'GLE', 'LPE', 'LLE')) + ' |', '',
                 'This is a TUS-REC2024 adaptation of the public MoGLo-Net implementation.',
                 'The public model uses five-frame sequences, shared encoders, patch correlation, global-local attention, recurrent motion heads, and MME/correlation/triplet losses.',
                 'The paper data protocol and TUS-REC2024 subject split differ; these results are not the paper table values.',
                 'All 72 public validation scans and original-resolution pixels were evaluated.',
                 'Raw records include environment, source hashes, commands, logs, steps.jsonl, history.json, loss_curves.png/svg, per-scan predictions, references, errors and per_scan.jsonl.']
        (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
        write_json(out / 'comparison.json', {'moglo_net': metrics})
        checksums = {}
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.name not in ('artifacts_sha256.json', 'status.json', 'gpu_telemetry.jsonl'):
                checksums[str(path.relative_to(out))] = sha256(path)
        write_json(out / 'artifacts_sha256.json', checksums)
        print(json.dumps({'output': str(out), 'metrics': metrics['mean']}), flush=True)
    except BaseException as exc:
        update(out, method, state='failed', error=str(exc), traceback=traceback.format_exc())
        update(out, 'experiment', state='failed', error=str(exc))
        raise
    finally:
        done.set()
        if monitor is not None:
            monitor.join(timeout=10)


if __name__ == '__main__':
    main()
