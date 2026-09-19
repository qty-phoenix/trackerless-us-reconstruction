#!/usr/bin/env python3
"""Print live download sizes and verified split readiness; no third-party dependencies."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'data/tus-rec2024'
for key in ['train1', 'train2']:
    record = json.loads((root / 'metadata' / f'zenodo_{key}.json').read_text())
    entry = next(x for x in record['files'] if x['key'].endswith('.zip'))
    complete = root / 'archives' / entry['key']
    partial = complete.with_suffix('.zip.part')
    path = complete if complete.exists() else partial
    size = path.stat().st_size if path.exists() else 0
    print(f'{entry["key"]}: {size/1e9:.3f}/{entry["size"]/1e9:.3f} GB ({size/entry["size"]:.2%})')
status = root / 'status.json'
if status.exists():
    print(status.read_text())
error = root / 'prepare_error.json'
if error.exists():
    print('Preparation error (check timestamp):', error.read_text())
