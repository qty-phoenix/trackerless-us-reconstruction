#!/usr/bin/env python3
"""Run the FairBench-v1 method queue, one experiment per GPU."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import threading


REPO = Path(__file__).resolve().parents[1]
LOCK = threading.Lock()
LANES = {
    0: ('two_frame_cnn', 'moglo_net', 'longterm'),
    1: ('dc2_net', 'nr_rec_fus', 'recon'),
}
METHODS = tuple(dict.fromkeys(method for methods in LANES.values() for method in methods))


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_status(status_path, key, **values):
    with LOCK:
        status = json.loads(status_path.read_text()) if status_path.exists() else {
            'protocol': 'FairBench-v1', 'started_utc': utc()}
        status.setdefault('jobs', {}).setdefault(key, {}).update(values, updated_utc=utc())
        temporary = status_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(status, indent=2) + '\n')
        temporary.replace(status_path)


def lane(root, queue_status_path, gpu, methods, seeds, resume_incomplete):
    log_path = root / f'queue_gpu{gpu}.log'
    with log_path.open('a', buffering=1) as log:
        for seed in seeds:
            for method in methods:
                key = f'seed_{seed}/{method}'
                output = root / f'seed_{seed}' / method
                output.parent.mkdir(parents=True, exist_ok=True)
                experiment_status_path = output / 'status.json'
                if experiment_status_path.exists():
                    status = json.loads(experiment_status_path.read_text())
                    if status.get('experiment', {}).get('state') == 'complete':
                        write_status(queue_status_path, key, state='complete', skipped_existing=True, gpu=gpu)
                        continue
                    if not resume_incomplete:
                        raise RuntimeError(f'Refusing to overwrite incomplete experiment: {output}')
                argv = [sys.executable, '-u', 'scripts/run_fairbench_experiment.py',
                        '--method', method, '--output', str(output), '--gpu', str(gpu),
                        '--seed', str(seed)]
                if experiment_status_path.exists():
                    argv.append('--resume')
                write_status(queue_status_path, key, state='running', gpu=gpu, command=argv,
                             started_utc=utc())
                print(json.dumps({'event': 'start', 'job': key, 'gpu': gpu, 'utc': utc()}),
                      file=log, flush=True)
                code = subprocess.run(argv, cwd=REPO, stdout=log,
                                      stderr=subprocess.STDOUT).returncode
                state = 'complete' if code == 0 else 'failed'
                write_status(queue_status_path, key, state=state, gpu=gpu, exit_code=code,
                             ended_utc=utc())
                print(json.dumps({'event': state, 'job': key, 'gpu': gpu,
                                  'exit_code': code, 'utc': utc()}), file=log, flush=True)
                if code:
                    raise RuntimeError(f'{key} failed with exit code {code}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='runs/fairbench_v1')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--gpus', type=int, nargs='+', choices=sorted(LANES),
                        default=sorted(LANES))
    parser.add_argument('--methods', nargs='+', choices=METHODS,
                        help='Override the method lane; requires exactly one --gpus value')
    parser.add_argument('--resume-incomplete', action='store_true')
    parser.add_argument('--status-name', default='queue_status.json')
    args = parser.parse_args()
    if args.methods and len(args.gpus) != 1:
        parser.error('--methods requires exactly one GPU')
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    queue_status_path = root / args.status_name
    cache = REPO / 'data/tus-rec2024/preprocessed/two_frame_cnn_fairbench_120x160/metadata.json'
    if not cache.exists() or not json.loads(cache.read_text()).get('complete'):
        raise RuntimeError(f'FairBench frame cache is incomplete: {cache}')
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(lane, root, queue_status_path, gpu,
                               tuple(args.methods) if args.methods else LANES[gpu], args.seeds,
                               args.resume_incomplete) for gpu in args.gpus]
        for future in futures:
            future.result()
    status = json.loads(queue_status_path.read_text())
    status.update(state='complete', ended_utc=utc())
    temporary = queue_status_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(status, indent=2) + '\n')
    temporary.replace(queue_status_path)


if __name__ == '__main__':
    main()
