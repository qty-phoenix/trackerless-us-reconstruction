# MoGLo-Net

This is a TUS-REC2024 adaptation of the authors' public US3D implementation. It preserves the paper's main motion-learning design: shared feature encoding, patch correlation, global-local attention, recurrent local/global heads, and MME plus correlation plus triplet losses.

The adaptation changes the input protocol to five TUS frames at 120×160, derives labels from the TUS calibration and tracker transforms, and evaluates full scans with the official millimetre GP/GL/LP/LL geometry. It must not be reported as a numerical reproduction of the paper's Forearm_Main experiment.

The model never receives tracker transforms in its forward pass. Tracker transforms enter only the training objective and post-inference scorer. Every run writes resumable checkpoints, `history.json`, per-batch `steps.jsonl`, `loss_curves.png/svg`, fixed-window validation metrics and full-scan raw artifacts.
