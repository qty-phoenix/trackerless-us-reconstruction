# DC²-Net

This directory independently implements the method described in Guo et al., “Ultrasound Volume Reconstruction From Freehand Scans Without Tracking,” IEEE TBME 70(3), 2023, DOI `10.1109/TBME.2022.3206596`. No author source repository was linked by the paper or found in the public project indexes checked during this reproduction, so this must not be described as a source-exact reproduction.

## Preserved paper design

- Five consecutive frames form one video segment.
- A 3D ResNeXt-50 (32×4d) extracts temporal and spatial context.
- A sigmoid attention branch weights the final residual map before pooling.
- Each segment regresses the mean neighboring 6-DoF motion in millimetres and degrees.
- Training combines MSE, case-wise Pearson correlation and contrastive margin-ranking losses with paper weights `5:1:1` and margin `0.25`.
- Formal training uses 100 epochs, Adam-equivalent AdamW with zero weight decay, learning rate `1e-4`, and a factor `0.9` every five epochs.

The grouped loader returns four overlapping five-frame segments from one scan, allowing Pearson correlation to be computed within a case. Every epoch visits all 1,200 training scans once; validation uses three fixed groups for each of 72 scans (216 groups). The paper iterated every possible segment and used batch size 48. The TUS adaptation uses four segments per case and batch size two because a full 3D ResNeXt-50 does not fit the original batch on a 24 GiB RTX 3090. This deliberate budget difference is recorded in every checkpoint.

## TUS-REC2024 differences

The paper used 618 transrectal scans and 100 transperineal/abdominal scans, cropped a fan-shaped prostate ROI to 224×224, and reported distance/frame/drift/prostate metrics. Here subjects 000–049 train and 050–052 validate on the public forearm data. Linear-probe images remain uncropped and are resized to 120×160, matching the common repository protocol. Model selection uses fixed validation windows, and final evaluation covers all 72 scans, every 480×640 pixel, and GPE/GLE/LPE/LLE with an unweighted mean over scans.

At inference, all overlapping five-frame windows are evaluated. A window predicts the mean motion of its four edges; every edge receives the mean prediction of all windows that cover it. Adjacent rigid transforms are then composed. Only frames and calibration enter inference; tracked poses are opened later by the scorer.

## Commands

```bash
/data/qty/anaconda3/bin/python -m baselines.dc2_net.train --device cuda

/data/qty/anaconda3/bin/python evaluate.py \
  --checkpoint runs/dc2_net/best.pt --method dc2_net --device cuda \
  --output runs/eval_dc2_net
```

The full experiment runner stores environment, code/data hashes, logs, checkpoints, per-step history, PNG/SVG curves, and complete per-scan evaluation artifacts:

```bash
/data/qty/anaconda3/bin/python scripts/run_dc2_experiment.py \
  --output runs/experiments/dc2_full_gpu1 --gpu 1
```

For a CPU pipeline check (not a performance result):

```bash
/data/qty/anaconda3/bin/python -m baselines.dc2_net.train --device cpu \
  --epochs 1 --steps 1 --val-steps 1 --tiny-model \
  --output runs/dc2_smoke
```
