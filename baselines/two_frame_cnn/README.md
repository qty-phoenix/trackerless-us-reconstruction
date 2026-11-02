# 2-Frame CNN (Prevost et al., 2018)

This directory adapts the image-only CNN from Raphael Prevost et al., “3D
freehand ultrasound without external tracking using deep learning,” *Medical
Image Analysis* 48 (2018), 187–202, DOI
[`10.1016/j.media.2018.06.003`](https://doi.org/10.1016/j.media.2018.06.003),
to the TUS-REC2024 protocol used by this repository.

The paper did not release an official training repository.  The network
specification is independently implemented from the paper-style
`Prevost2018ConvNet` supplied by the official
[DualTrack](https://github.com/ImFusionGmbH/DualTrack) repository as a
third-party reproduction.  Consequently this is a method-faithful adaptation,
not a claim of source-exact or paper-number reproduction.

## Model and supervision

Two successive grayscale frames are stacked as a two-channel `128 x 128`
input.  Four stride-two convolutions, two max-pooling operations and two fully
connected layers regress `[tx, ty, tz, rx, ry, rz]`: image-coordinate
translation in millimetres and XYZ Euler rotation in degrees.  The loss is the
paper's direct six-parameter MSE.  Training labels are derived from tracked
poses, but inference receives only images and the public calibration.

The TUS adaptation follows DualTrack's published `epoch_mode=tus_rec` recipe:

- one deterministic random frame pair from each of 1,200 training scans per
  epoch;
- batch size 16, Adam at `1e-4`, no learning-rate scheduler, gradient clipping
  at 3 and dropout 0.25;
- horizontal flip probability 0.5, sequence reversal probability 0.5 and
  one-frame skipping probability 0.2;
- 6,700 scan-wise epochs, with three fixed validation pairs from each of 72
  validation scans every 100 epochs.

For I/O efficiency, the executable configuration packs 100 consecutive
scan-wise epochs into one stored epoch (`67 x 100 = 6,700`).  Each packed slot
uses the exact virtual-epoch pair RNG and scan permutation; batches never cross
a virtual-epoch boundary because 1,200 is divisible by 16.  Thus the run still
contains 8,040,000 training pairs and 502,500 Adam updates.  Only DataLoader,
checkpoint and history bookkeeping occur 67 rather than 6,700 times.

The DualTrack README's main TUS command substitutes EfficientNet-B1.  This
implementation deliberately selects its separate original shallow
`Prevost2018ConvNet` architecture so the result remains a classical 2018 CNN
baseline.  Its model factory explicitly constructs this network with
`custom_init=False`, so this implementation likewise retains PyTorch's default
layer initialization rather than activating the class's optional Gaussian
initializer. Horizontal flips are accompanied by the corresponding physical
image-coordinate reflection of the pose label.

## Train, resume and evaluate

```bash
# One-time, resumable 128 x 128 uint8 image plus float32 pose-label cache
# (about 10 GB). It changes only storage and I/O, not sampled pairs or labels.
/data/qty/anaconda3/bin/python scripts/preprocess_two_frame_cnn.py --device cuda

/data/qty/anaconda3/bin/python -m baselines.two_frame_cnn.train --device cuda

/data/qty/anaconda3/bin/python -m baselines.two_frame_cnn.train --device cuda \
  --resume runs/two_frame_cnn/last.pt

/data/qty/anaconda3/bin/python evaluate.py \
  --checkpoint runs/two_frame_cnn/best.pt --method two_frame_cnn --device cuda \
  --output runs/eval_two_frame_cnn
```

Full scans are reconstructed by predicting every adjacent transform and
composing them from the first frame.  The common evaluator reports GPE, GLE,
LPE and LLE over all 72 validation scans and all original-resolution pixels.

The durable experiment wrapper records the source/environment/data snapshot,
GPU telemetry, all batch logs, resumable checkpoints, curves and complete
per-scan evaluation artifacts:

```bash
/data/qty/anaconda3/bin/python scripts/run_two_frame_cnn_experiment.py \
  --output runs/experiments/two_frame_cnn_full --gpu 0
```

For a CPU plumbing check only (not a result):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/qty/anaconda3/bin/python \
  -m baselines.two_frame_cnn.train --device cpu --epochs 1 \
  --steps 1 --val-steps 1 --workers 0 \
  --output runs/two_frame_cnn_smoke
```

## Reproduction boundary

The original work used a private 800-sweep in-vivo dataset and reported drift
and length errors; it also studied optional IMU input.  This adaptation uses no
IMU, uses the public forearm TUS-REC2024 subject split (`000–049` train,
`050–052` validation), resizes the challenge images to the architecture's
square input and evaluates challenge DDF metrics.  Its numerical values must
therefore be compared only with the other methods retrained under this
repository's common protocol, not with the paper table.
