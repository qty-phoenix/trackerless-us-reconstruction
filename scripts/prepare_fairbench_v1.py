#!/usr/bin/env python3
"""Create immutable subject-disjoint manifests for the FairBench-v1 protocol."""
import hashlib
import json
from pathlib import Path
import shutil


REPO = Path(__file__).resolve().parents[1]


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def write_jsonl(path, values):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in values))
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data = REPO / 'data/tus-rec2024'
    output = data / 'preprocessed/fairbench_v1'
    output.mkdir(parents=True, exist_ok=True)
    official_train = rows(data / 'manifests/train.jsonl')
    official_val = rows(data / 'manifests/val.jsonl')
    definitions = {
        'train': (official_train, {f'{i:03d}' for i in range(0, 40)}, 960),
        'val': (official_train, {f'{i:03d}' for i in range(40, 45)}, 120),
        'internal_test': (official_train, {f'{i:03d}' for i in range(45, 50)}, 120),
        'external_val': (official_val, {f'{i:03d}' for i in range(50, 53)}, 72),
    }
    metadata = {'protocol': 'FairBench-v1',
                'roles': {'train': 'optimization', 'val': 'checkpoint selection',
                          'internal_test': 'held-out internal test',
                          'external_val': 'public official validation; not organizer-held test'},
                'organizer_test': {'subjects': 32, 'scans': 768, 'available': False},
                'splits': {}}
    seen = set()
    for name, (source, subjects, expected) in definitions.items():
        selected = [row for row in source if row['subject'] in subjects]
        counts = {subject: sum(row['subject'] == subject for row in selected)
                  for subject in sorted(subjects)}
        if len(selected) != expected or set(counts.values()) != {24}:
            raise ValueError(f'{name}: expected {expected} scans and 24 per subject, got {counts}')
        if seen & subjects:
            raise ValueError(f'Subject overlap in {name}: {sorted(seen & subjects)}')
        seen.update(subjects)
        manifest = output / f'{name}.jsonl'
        write_jsonl(manifest, selected)
        metadata['splits'][name] = {'subjects': sorted(subjects), 'scans': len(selected),
                                    'frames': sum(int(row['num_frames']) for row in selected),
                                    'manifest': str(manifest.relative_to(REPO)),
                                    'sha256': sha256(manifest)}
    source_preprocessed = data / 'preprocessed/longterm'
    shutil.copy2(source_preprocessed / 'image_to_tool_downsampled.npy',
                 output / 'image_to_tool_downsampled.npy')
    write_json(output / 'metadata.json', metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
