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
methods = tuple(name for name in ('longterm', 'nr_rec_fus', 'moglo_net', 'recon', 'dc2_net',
                                  'two_frame_cnn')
                if name in status or (a.directory / name).exists())
result = {'experiment': str(a.directory), 'state': status.get('experiment', {}).get('state')}
for method in methods:
    folder = a.directory / method
    record = {k: v for k, v in status.get(method, {}).items()
              if k in ('state', 'phase', 'gpu', 'error', 'metrics')}
    history_candidates = ([('backbone', folder / 'backbone/history.json'),
                           ('discriminator', folder / 'training/history.json')]
                          if method == 'recon' else
                          [('training', folder / 'training/history.json'),
                           ('training', folder / 'history.json')])
    existing_histories = []
    for stage, history in history_candidates:
        if history.exists():
            values = json.loads(history.read_text())
            existing_histories.append((stage, history, values))
            record[f'{stage}_epochs'] = len(values)
            record[f'{stage}_last_epoch'] = values[-1]
            selection = [v['val_distance_mm'] for v in values if 'val_distance_mm' in v]
            if not selection:
                selection = [v['val_loss'] for v in values if 'val_loss' in v]
            if selection:
                record[f'{stage}_best_selection'] = min(selection)
    if existing_histories:
        stage, history, values = existing_histories[-1]
        record['completed_epochs'] = len(values)
        record['last_epoch'] = values[-1]
        selection = [v['val_distance_mm'] for v in values if 'val_distance_mm' in v]
        if not selection:
            selection = [v['val_loss'] for v in values if 'val_loss' in v]
        if selection:
            record['best_selection'] = min(selection)
    raw_candidates = ([folder / 'training/steps.jsonl', folder / 'backbone/steps.jsonl']
                      if method == 'recon' else
                      [folder / 'training/steps.jsonl', folder / 'steps.jsonl'])
    raw = next((path for path in raw_candidates if path.exists()), raw_candidates[0])
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
if any(result[m].get('state') == 'running' for m in methods):
    result['state'] = 'running'
if a.brief:
    for method in methods:
        row = result[method]
        last = row.get('last_step', {})
        result[method] = {'state': row.get('state'), 'phase': row.get('phase'),
                          'epochs_done': row.get('completed_epochs'),
                          'last_val_mm': row.get('last_epoch', {}).get('val_distance_mm'),
                          'best_selection': row.get('best_selection'),
                          'backbone_epochs': row.get('backbone_epochs'),
                          'discriminator_epochs': row.get('discriminator_epochs'),
                          'current_epoch': last.get('epoch'), 'split': last.get('split'),
                          'step': last.get('step'), 'utc': last.get('utc'),
                          'eval_scans': row.get('evaluation_scans'),
                          'rigid_eval_scans': row.get('evaluation_rigid_scans')}
print(json.dumps(result, ensure_ascii=False, indent=2))
