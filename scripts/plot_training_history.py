#!/usr/bin/env python3
"""Create the standard loss plots from an existing history.json."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training import plot_history


parser = argparse.ArgumentParser()
parser.add_argument('history', type=Path)
parser.add_argument('--output', type=Path, default=None)
parser.add_argument('--title', default='Training history')
args = parser.parse_args()
history = json.loads(args.history.read_text())
plot_history(history, args.output or args.history.parent, args.title)
print(f'Wrote {(args.output or args.history.parent) / "loss_curves.png"}')
