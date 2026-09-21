#!/usr/bin/env python3
"""Read experiment progress without touching the running GPU processes."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('directory', type=Path)
p.add_argument('--brief', action='store_true')
a = p.parse_args()
status = json.loads((a.directory / 'status.json').read_text())
result = {'experiment': str(a.directory), 'state': status.get('experiment', {}).get('state')}
for method in ('longterm', 'nr_rec_fus'):
    folder = a.directory / method
    record = {k: v for k, v in status.get(method, {}).items()
              if k in ('state', 'phase', 'gpu', 'error', 'metrics')}
    history = folder / 'training/history.json'
    if history.exists():
        values = json.loads(history.read_text())
        record['completed_epochs'] = len(values)
        record['last_epoch'] = values[-1]
        record['best_val_distance_mm'] = min(v['val_distance_mm'] for v in values)
    raw = folder / 'training/steps.jsonl'
    if raw.exists() and raw.stat().st_size:
        with raw.open('rb') as stream:
            stream.seek(max(0, raw.stat().st_size - 16384))
            lines = stream.read().decode(errors='replace').splitlines()
        for line in reversed(lines):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'split' in entry:
                record['last_step'] = {k: entry[k] for k in ('utc', 'epoch', 'split', 'step', 'metrics')}
                break
    for name in ('evaluation', 'evaluation_rigid'):
        raw = folder / name / 'per_scan.jsonl'
        if raw.exists():
            lines = raw.read_text().splitlines()
            record[name + '_scans'] = len(lines)
    result[method] = record
if any(result[m].get('state') == 'running' for m in ('longterm', 'nr_rec_fus')):
    result['state'] = 'running'
if a.brief:
    for method in ('longterm', 'nr_rec_fus'):
        row = result[method]
        last = row.get('last_step', {})
        result[method] = {'state': row.get('state'), 'phase': row.get('phase'),
                          'epochs_done': row.get('completed_epochs'),
                          'last_val_mm': row.get('last_epoch', {}).get('val_distance_mm'),
                          'best_val_mm': row.get('best_val_distance_mm'),
                          'current_epoch': last.get('epoch'), 'split': last.get('split'),
                          'step': last.get('step'), 'utc': last.get('utc'),
                          'eval_scans': row.get('evaluation_scans'),
                          'rigid_eval_scans': row.get('evaluation_rigid_scans')}
print(json.dumps(result, ensure_ascii=False, indent=2))
