#!/usr/bin/env python3
"""Train and fully evaluate the DC2-Net TUS-REC2024 adaptation."""
import argparse
import json
from pathlib import Path
import threading
import traceback

from run_baseline_experiment import (REPO, command, evaluate, sha256, snapshot,
                                     telemetry, update, utc, write_json)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--config', default='baselines/dc2_net/configs/tus_rec2024.json')
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    method = 'dc2_net'
    folder = out / method
    folder.mkdir()
    update(out, 'experiment', state='initializing')
    done, monitor = threading.Event(), None
    try:
        snapshot(out, min_gpus=2)
        config = json.loads((REPO / args.config).read_text())
        config['output'] = str(folder / 'training')
        config_path = folder / 'train_config.json'
        write_json(config_path, config)
        monitor = threading.Thread(target=telemetry, args=(out, done), daemon=True)
        monitor.start()
        command(out, method, args.gpu, 'training', [
            '-m', 'baselines.dc2_net.train', '--config', str(config_path),
            '--device', 'cuda', '--workers', '4'])
        checkpoint = folder / 'training/best.pt'
        history = json.loads((folder / 'training/history.json').read_text())
        if (len(history) != config['epochs'] or
                any(row['train_windows'] != 1200 or row['val_windows'] != 216
                    for row in history)):
            raise RuntimeError('DC2-Net history does not cover all configured case groups')
        metrics = evaluate(out, method, args.gpu, checkpoint)
        update(out, method, state='complete', phase='complete', gpu=args.gpu,
               checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
               metrics=metrics['mean'], ended_utc=utc())
        update(out, 'experiment', state='complete', ended_utc=utc())
        lines = ['# DC2-Net TUS-REC2024 experiment', '', f'Completed UTC: {utc()}', '',
                 '| Method | GPE (mm) | GLE (mm) | LPE (mm) | LLE (mm) |',
                 '| --- | ---: | ---: | ---: | ---: |',
                 '| dc2_net | ' + ' | '.join(f"{metrics['mean'][key]:.6f}"
                                               for key in ('GPE', 'GLE', 'LPE', 'LLE')) + ' |', '',
                 'This is an independent TUS-REC2024 adaptation of the published DC2-Net method; no author source release was available.',
                 'It preserves five-frame 3D ResNeXt-50 context, attention, mean 6-DoF regression, case-wise correlation and margin-ranking losses.',
                 'The prostate paper datasets, crop, exhaustive-window batch protocol and metrics differ from this forearm adaptation.',
                 'All 72 public validation scans and every original-resolution pixel were evaluated.',
                 'Raw records include environment/source/data hashes, commands, logs, steps, histories, curves and per-scan predictions/references/errors.']
        (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
        write_json(out / 'comparison.json', {'dc2_net': metrics})
        checksums = {}
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.name not in ('artifacts_sha256.json', 'status.json',
                                                    'gpu_telemetry.jsonl'):
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
