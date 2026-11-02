# FairBench-v1 controlled comparison

FairBench-v1 is separate from the paper-faithful reproduction track. It keeps
each method's architecture, native temporal context, core loss and inference
algorithm, while fixing external data and optimization conditions.

## Data roles

| Role | Subjects | Scans |
| --- | --- | ---: |
| Training | 000–039 | 960 |
| Checkpoint selection | 040–044 | 120 |
| Internal test | 045–049 | 120 |
| External public validation | 050–052 | 72 |

The organizer-held test set contains another 32 subjects and 768 scans but is
not available locally. The public 050–052 set is therefore never described as
the official test set.

All methods read the same cached `120×160` uint8 frames and float32 poses.
Training and selection subjects never overlap, and neither test role enters
training or checkpoint selection.

## Fixed optimization budget

- 450 total method-native epochs, or 360 backbone plus 90 discriminator epochs
  for RecON.
- 960 training items per epoch.
- Physical micro-batches are method-dependent, but gradient accumulation fixes
  the effective batch at eight items.
- 120 optimizer updates per epoch, 54,000 total updates, and 432,000 total
  method-native items.
- Adam at `1e-4`, zero weight decay, no learning-rate schedule, gradient clip
  three, no external pretrained weights.
- Validation every 25 epochs on 16 fixed windows per selection scan.
- Best checkpoints use the same four-corner millimetre error of the first
  method-native motion primitive. DC2-Net's primitive remains its paper-defined
  mean motion over a five-frame segment.
- Formal seeds are 0, 1 and 2.

One item contains the native context required by a method (two frames for
2-Frame CNN, four for NR-Rec-FUS, five for MoGLo/DC2-Net, ten for Long-Term and
sixteen for RecON). Consequently this is item/update matched, not frame-, FLOP-
or wall-clock-matched. Parameters, runtime and memory are retained as secondary
efficiency measurements.

## Evaluation

Every selected checkpoint is evaluated on all 120 internal-test scans and all
72 external-validation scans with the same original-resolution geometry and
unweighted scan aggregation. Primary NR-Rec-FUS metrics use its rigid branch;
native refined metrics are additional. RecON reports both its canonical
30-epoch label-free online adaptation and an offline-only result.

The durable two-GPU queue is:

```bash
/data/qty/anaconda3/bin/python -u scripts/run_fairbench_queue.py \
  --output runs/fairbench_v1_formal --seeds 0 1 2
```

GPU 0 runs 2-Frame CNN, MoGLo-Net and Long-Term sequentially. GPU 1 runs
DC2-Net, NR-Rec-FUS and RecON sequentially. Each experiment snapshots source,
environment and protocol hashes and stores training steps, best/last
checkpoints, full per-scan evaluations and artifact checksums.
