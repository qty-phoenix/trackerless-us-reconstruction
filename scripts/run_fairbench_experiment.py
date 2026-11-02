#!/usr/bin/env python3
"""Train one method and evaluate both FairBench-v1 holdouts."""
import argparse
import json
from pathlib import Path
import threading
import traceback

from run_baseline_experiment import (REPO, command, sha256, snapshot, telemetry,
                                     update, utc, write_json)


METHODS = ('longterm', 'nr_rec_fus', 'moglo_net', 'recon', 'dc2_net',
           'two_frame_cnn')
CONFIGS = {
    name: REPO / f'baselines/{name}/configs/fairbench_v1.json'
    for name in METHODS if name != 'longterm'
}
FAIR = REPO / 'data/tus-rec2024/preprocessed/fairbench_v1'


def load_history(path, expected_epochs, allow_partial=False):
    history = json.loads(path.read_text())
    valid_length = (len(history) == expected_epochs or
                    (allow_partial and 0 < len(history) < expected_epochs))
    if not valid_length:
        raise RuntimeError(f'{path}: expected {expected_epochs} epochs, found {len(history)}')
    if [row.get('epoch') for row in history] != list(range(len(history))):
        raise RuntimeError(f'{path}: epochs are not contiguous from zero')
    if any(row.get('train_windows') != 960 or row.get('optimizer_steps') != 120
           for row in history):
        raise RuntimeError(f'{path}: training did not preserve 960 items and 120 updates per epoch')
    return history


def evaluate_manifest(out, method, gpu, checkpoint, name, extra=(), tag=''):
    manifest = FAIR / f'{name}.jsonl'
    suffix = name + (('_' + tag) if tag else '')
    output = out / method / suffix
    command(out, method, gpu, suffix, [
        'evaluate.py', '--checkpoint', str(checkpoint), '--method', method,
        '--device', 'cuda', '--cpu-threads', '4', '--chunk-size', '131072',
        '--manifest', str(manifest), '--output', str(output), *extra])
    metrics = json.loads((output / 'metrics.json').read_text())
    expected = 120 if name == 'internal_test' else 72
    if metrics['partial'] or metrics['scan_count'] != expected:
        raise RuntimeError(f'{method} {suffix}: incomplete evaluation')
    return metrics


def recover_training(training, expected_epochs, resume):
    history = training / 'history.json'
    if not resume or not training.exists():
        return None, None
    best, last = training / 'best.pt', training / 'last.pt'
    if not history.exists() or not best.exists():
        raise RuntimeError(f'Cannot safely resume incomplete training directory: {training}')
    rows = load_history(history, expected_epochs, allow_partial=True)
    if len(rows) == expected_epochs:
        return best, None
    if not last.exists():
        raise RuntimeError(f'Incomplete training has no last.pt checkpoint: {training}')
    return None, last


def train(out, method, gpu, seed, resume=False):
    folder = out / method
    if method == 'longterm':
        training = folder / 'training'
        checkpoint, resume_checkpoint = recover_training(training, 450, resume)
        if checkpoint is not None:
            return checkpoint
        training_args = [
            'train_longterm.py', '--root', 'data/tus-rec2024',
            '--preprocessed', str(FAIR), '--output', str(training),
            '--frame-cache', 'data/tus-rec2024/preprocessed/two_frame_cnn_fairbench_120x160',
            '--epochs', '450', '--batch-size', '8', '--effective-batch-size', '8',
            '--val-batch-size', '8', '--workers', '4', '--num-samples', '10',
            '--sample-range', '10', '--val-windows', '16', '--lr', '0.0001',
            '--optimizer', 'adam', '--weight-decay', '0', '--grad-clip', '3',
            '--validate-every', '25', '--plot-every', '25', '--seed', str(seed),
            '--device', 'cuda']
        if resume_checkpoint is not None:
            training_args += ['--resume', str(resume_checkpoint)]
        command(out, method, gpu, 'training', training_args)
        load_history(training / 'history.json', 450)
        return training / 'best.pt'

    config = json.loads(CONFIGS[method].read_text())
    config['seed'] = seed
    if method == 'recon':
        backbone = folder / 'backbone'
        discriminator = folder / 'discriminator'
        config.update(backbone_output=str(backbone), output=str(discriminator))
        config_path = folder / 'train_config.json'
        write_json(config_path, config)
        checkpoint, discriminator_resume = recover_training(
            discriminator, config['discriminator_epochs'], resume)
        if checkpoint is not None:
            return checkpoint
        backbone_checkpoint, backbone_resume = recover_training(
            backbone, config['backbone_epochs'], resume)
        if backbone_checkpoint is None:
            backbone_args = [
                '-m', 'baselines.recon.train', '--stage', 'backbone', '--config',
                str(config_path), '--device', 'cuda', '--workers', '4',
                '--no-imagenet-pretrained']
            if backbone_resume is not None:
                backbone_args += ['--resume', str(backbone_resume)]
            command(out, method, gpu, 'backbone_training', backbone_args)
            load_history(backbone / 'history.json', config['backbone_epochs'])
            backbone_checkpoint = backbone / 'best.pt'
        discriminator_args = [
            '-m', 'baselines.recon.train', '--stage', 'discriminator', '--config',
            str(config_path), '--backbone-checkpoint', str(backbone_checkpoint),
            '--device', 'cuda', '--workers', '4']
        if discriminator_resume is not None:
            discriminator_args += ['--resume', str(discriminator_resume)]
        command(out, method, gpu, 'discriminator_training', discriminator_args)
        load_history(discriminator / 'history.json', config['discriminator_epochs'])
        return discriminator / 'best.pt'

    training = folder / 'training'
    checkpoint, resume_checkpoint = recover_training(training, config['epochs'], resume)
    if checkpoint is not None:
        return checkpoint
    config['output'] = str(training)
    config_path = folder / 'train_config.json'
    write_json(config_path, config)
    modules = {'nr_rec_fus': 'baselines.nr_rec_fus.train',
               'moglo_net': 'baselines.moglo_net.train',
               'dc2_net': 'baselines.dc2_net.train',
               'two_frame_cnn': 'baselines.two_frame_cnn.train'}
    training_args = [
        '-m', modules[method], '--config', str(config_path), '--device', 'cuda',
        '--workers', '4']
    if resume_checkpoint is not None:
        training_args += ['--resume', str(resume_checkpoint)]
    command(out, method, gpu, 'training', training_args)
    load_history(training / 'history.json', config['epochs'])
    return training / 'best.pt'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--resume', action='store_true',
                        help='Reuse a complete existing training stage and finish missing evaluation')
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    folder = out / args.method
    folder.mkdir(exist_ok=args.resume)
    update(out, 'experiment', state='resuming' if args.resume else 'initializing',
           protocol='FairBench-v1', seed=args.seed, error=None, traceback=None)
    if args.resume:
        update(out, args.method, state='resuming', error=None, traceback=None)
    done, monitor = threading.Event(), None
    try:
        if not args.resume:
            snapshot(out, min_gpus=2)
            write_json(out / 'protocol.json', json.loads((FAIR / 'metadata.json').read_text()))
        else:
            resume_snapshot = out / 'resume_snapshot'
            resume_snapshot.mkdir(exist_ok=True)
            snapshot(resume_snapshot, min_gpus=2)
        monitor = threading.Thread(target=telemetry, args=(out, done), daemon=True)
        monitor.start()
        checkpoint = train(out, args.method, args.gpu, args.seed, args.resume)
        primary_extra = ('--rigid-only',) if args.method == 'nr_rec_fus' else ()
        results = {
            'internal_test': evaluate_manifest(out, args.method, args.gpu, checkpoint,
                                               'internal_test', primary_extra),
            'external_val': evaluate_manifest(out, args.method, args.gpu, checkpoint,
                                              'external_val', primary_extra)}
        if args.method == 'nr_rec_fus':
            results['internal_test_native'] = evaluate_manifest(
                out, args.method, args.gpu, checkpoint, 'internal_test', tag='native')
            results['external_val_native'] = evaluate_manifest(
                out, args.method, args.gpu, checkpoint, 'external_val', tag='native')
        if args.method == 'recon':
            results['internal_test_offline'] = evaluate_manifest(
                out, args.method, args.gpu, checkpoint, 'internal_test',
                ('--online-epochs', '0'), 'offline')
            results['external_val_offline'] = evaluate_manifest(
                out, args.method, args.gpu, checkpoint, 'external_val',
                ('--online-epochs', '0'), 'offline')
        write_json(out / 'comparison.json', results)
        update(out, args.method, state='complete', phase='complete', gpu=args.gpu,
               checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
               internal_test=results['internal_test']['mean'],
               external_val=results['external_val']['mean'], ended_utc=utc(),
               error=None, traceback=None)
        update(out, 'experiment', state='complete', ended_utc=utc(),
               error=None, traceback=None)
        lines = [f'# FairBench-v1: {args.method}, seed {args.seed}', '',
                 'Subject-disjoint train/selection/internal/external split: 000–039 / 040–044 / 045–049 / 050–052.',
                 'Training budget: 432,000 method-native items, effective batch 8, 54,000 optimizer updates.',
                 'All results cover complete full-resolution scans.', '']
        for name, metrics in results.items():
            mean = metrics['mean']
            lines += [f'## {name}', '',
                      '| GPE | GLE | LPE | LLE |', '| ---: | ---: | ---: | ---: |',
                      '| ' + ' | '.join(f"{mean[key]:.6f}" for key in
                                         ('GPE', 'GLE', 'LPE', 'LLE')) + ' |', '']
        (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
        checksums = {}
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.name not in ('artifacts_sha256.json', 'status.json',
                                                    'gpu_telemetry.jsonl'):
                checksums[str(path.relative_to(out))] = sha256(path)
        write_json(out / 'artifacts_sha256.json', checksums)
        print(json.dumps({'output': str(out), 'results': {k: v['mean'] for k, v in results.items()}}),
              flush=True)
    except BaseException as exc:
        update(out, args.method, state='failed', error=str(exc), traceback=traceback.format_exc())
        update(out, 'experiment', state='failed', error=str(exc))
        raise
    finally:
        done.set()
        if monitor is not None:
            monitor.join(timeout=10)


if __name__ == '__main__':
    main()
