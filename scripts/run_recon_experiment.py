#!/usr/bin/env python3
"""Train both RecON stages and run the complete TUS-REC2024 evaluation."""
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
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--config', default='baselines/recon/configs/tus_rec2024.json')
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    method = 'recon'
    folder = out / method
    folder.mkdir()
    update(out, 'experiment', state='initializing')
    done, monitor = threading.Event(), None
    try:
        snapshot(out, min_gpus=1)
        config = json.loads((REPO / args.config).read_text())
        config_path = folder / 'train_config.json'
        write_json(config_path, config)
        monitor = threading.Thread(target=telemetry, args=(out, done), daemon=True)
        monitor.start()
        backbone = folder / 'backbone'
        command(out, method, args.gpu, 'backbone_training', [
            '-m', 'baselines.recon.train', '--config', str(config_path),
            '--stage', 'backbone', '--device', 'cuda', '--workers', '4',
            '--output', str(backbone)])
        backbone_history = json.loads((backbone / 'history.json').read_text())
        if (len(backbone_history) != config['backbone_epochs'] or
                any(row['train_windows'] != 1200 or row['val_windows'] != 216
                    for row in backbone_history)):
            raise RuntimeError('RecON backbone history does not cover all configured windows')
        discriminator = folder / 'training'
        command(out, method, args.gpu, 'discriminator_training', [
            '-m', 'baselines.recon.train', '--config', str(config_path),
            '--stage', 'discriminator', '--backbone-checkpoint', str(backbone / 'best.pt'),
            '--device', 'cuda', '--workers', '4', '--output', str(discriminator)])
        discriminator_history = json.loads((discriminator / 'history.json').read_text())
        if (len(discriminator_history) != config['discriminator_epochs'] or
                any(row['train_windows'] != 1200 or row['val_windows'] != 216
                    for row in discriminator_history)):
            raise RuntimeError('RecON discriminator history does not cover all configured windows')
        checkpoint = discriminator / 'best.pt'
        metrics = evaluate(out, method, args.gpu, checkpoint)
        update(out, method, state='complete', phase='complete', checkpoint=str(checkpoint),
               checkpoint_sha256=sha256(checkpoint), metrics=metrics['mean'], ended_utc=utc())
        update(out, 'experiment', state='complete', ended_utc=utc())
        lines = ['# RecON TUS-REC2024 experiment', '', f'Completed UTC: {utc()}', '',
                 '| Method | GPE (mm) | GLE (mm) | LPE (mm) | LLE (mm) |',
                 '| --- | ---: | ---: | ---: | ---: |',
                 '| recon | ' + ' | '.join(f"{metrics['mean'][key]:.6f}"
                                            for key in ('GPE', 'GLE', 'LPE', 'LLE')) + ' |', '',
                 f"Backbone training completed {config['backbone_epochs']} epochs and discriminator training completed {config['discriminator_epochs']} epochs.",
                 f"Each scan used {config['online_epochs']} label-free online optimization epochs from the same initial checkpoint.",
                 'All 72 public validation scans and every original-resolution pixel were evaluated.',
                 'This is a TUS-REC2024 adaptation of RecON, not a reproduction of the paper table values.',
                 'Raw records include environment/source hashes, commands, logs, training steps, curves, embedded PSC provenance, and per-scan predictions/references/errors.']
        (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
        write_json(out / 'comparison.json', {'recon': metrics})
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
