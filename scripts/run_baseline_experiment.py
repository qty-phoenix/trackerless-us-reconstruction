#!/usr/bin/env python3
"""Durable two-GPU experiment: reuse trained Long-Term, train NR, evaluate both.

Run from the repository root. Each subprocess has a dedicated GPU and raw log.
The experiment directory is never reused or overwritten.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
LOCK = threading.Lock()


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def capture(command):
    return subprocess.check_output(command, cwd=REPO, text=True).strip()


def snapshot(out):
    import torch
    import torchvision
    import numpy
    import h5py
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError('Both CUDA GPUs must be visible to the experiment supervisor')
    versions = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
                'torch': torch.__version__, 'torchvision': torchvision.__version__,
                'numpy': numpy.__version__, 'h5py': h5py.__version__, 'cuda': torch.version.cuda,
                'cudnn': torch.backends.cudnn.version(), 'gpu_count': torch.cuda.device_count(),
                'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                'nvidia_smi': capture(['nvidia-smi']), 'git_head': capture(['git', 'rev-parse', 'HEAD']),
                'git_status': capture(['git', 'status', '--short']), 'utc': utc()}
    write_json(out / 'environment.json', versions)
    (out / 'code.patch').write_text(capture(['git', 'diff', 'HEAD']) + '\n')
    sources = capture(['git', 'ls-files', '--cached', '--others', '--exclude-standard']).splitlines()
    source_hashes = {}
    for name in sources:
        source = REPO / name
        if source.suffix not in ('.py', '.json', '.md', '.txt') and name != '.gitignore':
            continue
        if not source.is_file():
            continue
        target = out / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_hashes[name] = sha256(target)
    write_json(out / 'source_sha256.json', source_hashes)
    metadata = ['calib_matrix.csv', 'status.json', 'manifests/train.jsonl', 'manifests/val.jsonl',
                'preprocessed/longterm/train.jsonl', 'preprocessed/longterm/val.jsonl',
                'preprocessed/longterm/metadata.json', 'preprocessed/longterm/image_to_tool_downsampled.npy']
    hashes, files = {}, {}
    for name in metadata:
        source = REPO / 'data/tus-rec2024' / name
        if source.exists():
            target = out / 'data_metadata' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            hashes[name] = sha256(target)
    for split in ('train', 'val'):
        for line in (REPO / 'data/tus-rec2024/manifests' / f'{split}.jsonl').read_text().splitlines():
            row = json.loads(line)
            for key in ('frames_path', 'tforms_path', 'landmarks_path'):
                name = row[key]
                if name not in files:
                    source = REPO / 'data/tus-rec2024' / name
                    stat = source.stat()
                    files[name] = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    write_json(out / 'data_metadata/metadata_sha256.json', hashes)
    write_json(out / 'data_metadata/source_file_inventory.json', files)


def update(out, method, **values):
    with LOCK:
        path = out / 'status.json'
        status = json.loads(path.read_text()) if path.exists() else {'started_utc': utc(), 'pid': os.getpid()}
        status.setdefault(method, {}).update(values, updated_utc=utc())
        write_json(path, status)


def command(out, method, gpu, phase, args):
    log_path = out / method / f'{phase}.log'
    argv = [sys.executable, '-u', *args]
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '4',
           'MKL_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1'}
    record = {'phase': phase, 'argv': argv, 'cwd': str(REPO), 'gpu': gpu,
              'started_utc': utc(), 'environment_overrides': {k: env[k] for k in
                  ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'PYTHONUNBUFFERED')}}
    with log_path.open('w', buffering=1) as log:
        process = subprocess.Popen(argv, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        update(out, method, state='running', phase=phase, subprocess_pid=process.pid,
               log=str(log_path), gpu=gpu, command=argv)
        started = time.monotonic()
        code = process.wait()
    record.update(exit_code=code, ended_utc=utc(), elapsed_seconds=time.monotonic() - started)
    with (out / method / 'commands.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')
    if code:
        raise RuntimeError(f'{method} {phase} exited {code}; see {log_path}')


def evaluate(out, method, gpu, checkpoint, suffix='evaluation', rigid=False):
    args = ['evaluate.py', '--checkpoint', str(checkpoint), '--device', 'cuda', '--cpu-threads', '4',
            '--chunk-size', '131072', '--output', str(out / method / suffix)]
    if rigid:
        args.append('--rigid-only')
    command(out, method, gpu, suffix, args)
    result = json.loads((out / method / suffix / 'metrics.json').read_text())
    if result['partial'] or result['scan_count'] != 72:
        raise RuntimeError(f'{method}: evaluation is incomplete')
    return result


def worker(out, method, gpu):
    try:
        folder = out / method
        folder.mkdir()
        if method == 'longterm':
            old = REPO / 'runs/longterm'
            for name in ('best.pt', 'last.pt', 'history.json'):
                shutil.copy2(old / name, folder / name)
            history = json.loads((folder / 'history.json').read_text())
            if len(history) != 100 or history[-1]['epoch'] != 99:
                raise RuntimeError('Existing Long-Term did not complete 100 epochs')
            write_json(folder / 'reuse.json', {'source': str(old), 'reason': '100 training epochs already completed',
                       'history_entries': len(history), 'selection_protocol': 'legacy random-window validation',
                       'historical_per_batch_logs': 'not available; preserved original epoch history without fabrication',
                       'best_sha256': sha256(folder / 'best.pt'), 'last_sha256': sha256(folder / 'last.pt')})
            subprocess.run([sys.executable, 'scripts/plot_training_history.py', str(folder / 'history.json'),
                            '--title', 'Long-Term training history'], cwd=REPO, check=True)
            checkpoint = folder / 'best.pt'
        else:
            config = json.loads((REPO / 'baselines/nr_rec_fus/configs/tus_rec2024.json').read_text())
            config['output'] = str(folder / 'training')
            config_path = folder / 'train_config.json'
            write_json(config_path, config)
            command(out, method, gpu, 'training', ['-m', 'baselines.nr_rec_fus.train',
                    '--config', str(config_path), '--device', 'cuda', '--workers', '0'])
            checkpoint = folder / 'training/best.pt'
            history = json.loads((folder / 'training/history.json').read_text())
            if len(history) != config['epochs'] or any(r['train_windows'] != 1200 or r['val_windows'] != 216 for r in history):
                raise RuntimeError('NR training history does not cover all train/validation windows')
        result = evaluate(out, method, gpu, checkpoint)
        if method == 'nr_rec_fus':
            evaluate(out, method, gpu, checkpoint, 'evaluation_rigid', rigid=True)
        update(out, method, state='complete', phase='complete', metrics=result['mean'], checkpoint=str(checkpoint),
               checkpoint_sha256=sha256(checkpoint), ended_utc=utc())
        return True
    except BaseException as exc:
        update(out, method, state='failed', error=str(exc), traceback=traceback.format_exc())
        return False


def telemetry(out, done):
    with (out / 'gpu_telemetry.jsonl').open('a', buffering=1) as stream:
        while not done.is_set():
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                                     '--format=csv,noheader,nounits'], capture_output=True, text=True)
            stream.write(json.dumps({'utc': utc(), 'exit_code': result.returncode, 'csv': result.stdout.strip(),
                                     'stderr': result.stderr.strip()}) + '\n')
            done.wait(30)


def finish(out):
    results = {}
    rows = []
    for method in ('longterm', 'nr_rec_fus'):
        path = out / method / 'evaluation/metrics.json'
        if path.exists():
            results[method] = json.loads(path.read_text())
            for line in (path.parent / 'per_scan.jsonl').read_text().splitlines():
                rows.append({'method': method, **json.loads(line)})
    rigid = out / 'nr_rec_fus/evaluation_rigid/metrics.json'
    if rigid.exists():
        results['nr_rec_fus_rigid'] = json.loads(rigid.read_text())
    write_json(out / 'comparison.json', results)
    import csv
    if rows:
        with (out / 'per_scan_comparison.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ['# Baseline experiment', '', f'Completed UTC: {utc()}', '',
             '| Method | GPE (mm) | GLE (mm) | LPE (mm) | LLE (mm) |',
             '| --- | ---: | ---: | ---: | ---: |']
    for method, record in results.items():
        lines.append('| ' + method + ' | ' + ' | '.join(f"{record['mean'][k]:.6f}" for k in ('GPE', 'GLE', 'LPE', 'LLE')) + ' |')
    lines += ['', 'All reported evaluations cover the same 72 public validation scans and every original-resolution pixel.',
              'Long-Term reuses the existing 100-epoch checkpoint selected with legacy random windows. NR is a new 50-epoch adapted-model run selected with fixed windows; training budgets and selection protocols differ.',
              'The validation subjects are used for model selection, so these are not held-out challenge-test results.',
              'Raw records: commands.jsonl, *.log, training steps.jsonl, history.json and loss_curves.png/svg; per scan prediction.npz, reference.npz, errors.npz; per_scan.jsonl and per_scan_comparison.csv.',
              'prediction.npz stores transforms/calibration and NR fields/bounds, allowing all GP/GL/LP/LL values to be regenerated without storing hundreds of GB of dense arrays.',
              'Original Long-Term per-batch logs are unavailable; the original epoch history and both original checkpoints were preserved.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    checksums = {}
    for path in sorted(out.rglob('*')):
        if path.is_file() and path.name not in ('artifacts_sha256.json', 'status.json', 'supervisor.log', 'gpu_telemetry.jsonl'):
            checksums[str(path.relative_to(out))] = sha256(path)
    write_json(out / 'artifacts_sha256.json', checksums)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    update(out, 'experiment', state='initializing')
    try:
        snapshot(out)
        done = threading.Event()
        monitor = threading.Thread(target=telemetry, args=(out, done), daemon=True)
        monitor.start()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, out, method, gpu) for method, gpu in (('longterm', 0), ('nr_rec_fus', 1))]
            success = all([future.result() for future in futures])
        done.set()
        monitor.join(timeout=10)
        finish(out)
        update(out, 'experiment', state='complete' if success else 'failed', ended_utc=utc())
        print(json.dumps({'output': str(out), 'success': success}), flush=True)
        if not success:
            sys.exit(1)
    except BaseException as exc:
        update(out, 'experiment', state='failed', error=str(exc), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
