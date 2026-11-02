#!/usr/bin/env python3
"""Train and fully evaluate the Prevost 2018 two-frame CNN adaptation."""
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
    parser.add_argument('--config',
                        default='baselines/two_frame_cnn/configs/tus_rec2024.json')
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    method = 'two_frame_cnn'
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
            '-m', 'baselines.two_frame_cnn.train', '--config', str(config_path),
            '--device', 'cuda', '--workers', '8'])
        checkpoint = folder / 'training/best.pt'
        history = json.loads((folder / 'training/history.json').read_text())
        validation_rows = [row for row in history if 'val_windows' in row]
        expected_validations = (config['epochs'] + config['validate_every'] - 1) // config['validate_every']
        if (len(history) != config['epochs'] or
                any(row['train_windows'] != 1200 * config['pairs_per_scan']
                    for row in history) or
                len(validation_rows) != expected_validations or
                any(row['val_windows'] != 216 for row in validation_rows)):
            raise RuntimeError('2-Frame CNN history does not cover its configured scan-wise protocol')
        metrics = evaluate(out, method, args.gpu, checkpoint)
        update(out, method, state='complete', phase='complete', gpu=args.gpu,
               checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
               metrics=metrics['mean'], ended_utc=utc())
        update(out, 'experiment', state='complete', ended_utc=utc())
        equivalent_epochs = config.get(
            'equivalent_scanwise_epochs', config['epochs'] * config['pairs_per_scan'])
        training_pairs = config['epochs'] * 1200 * config['pairs_per_scan']
        experiment_kind = ('training-budget ablation' if config.get('ablation')
                           else 'experiment')
        lines = [f'# 2-Frame CNN TUS-REC2024 {experiment_kind}', '',
                 f'Completed UTC: {utc()}', '',
                 '| Method | GPE (mm) | GLE (mm) | LPE (mm) | LLE (mm) |',
                 '| --- | ---: | ---: | ---: | ---: |',
                 '| two_frame_cnn | ' + ' | '.join(
                     f"{metrics['mean'][key]:.6f}" for key in
                     ('GPE', 'GLE', 'LPE', 'LLE')) + ' |', '',
                 'This is a TUS-REC2024 adaptation of Prevost et al. 2018 using the shallow paper-style CNN exposed by the DualTrack third-party reproduction.',
                 'It is not an official-source or original-dataset numerical reproduction.',
                 f"The run used {training_pairs:,} training pairs across {equivalent_epochs:,} equivalent scan-wise epochs.",
                 'Inference composed every adjacent prediction.',
                 'All 72 public validation scans and every original-resolution pixel were evaluated.',
                 'Raw records contain environment/source/data hashes, commands, logs, batch histories, curves and per-scan predictions/references/errors.']
        if config.get('ablation'):
            lines.extend(['',
                          f"Ablation label: `{config['ablation']}`.",
                          'This run changes only the configured training budget; model, seed, optimizer, augmentation, validation and evaluation protocols are unchanged.'])
        (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
        write_json(out / 'comparison.json', {method: metrics})
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
