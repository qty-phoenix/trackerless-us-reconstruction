# NR-Rec-FUS

这是论文 [Nonrigid Reconstruction of Freehand Ultrasound Without a Tracker](https://arxiv.org/abs/2407.05767)（MICCAI 2024）官方实现思路在 TUS-REC2024 上的独立适配。目录保留刚性位姿分支、VoxelMorph 风格二维非刚性形变分支、数据适配和训练入口，避免覆盖 Long-Term 基线。

```text
nr_rec_fus/
├── configs/                 # TUS-REC2024 配置
├── preprocessing/           # 原地检查数据，不复制大文件
├── datasets/                # 窗口采样和 HDF5 读取
├── models/                  # rigid pose + deformation
├── train.py
└── results/
```

运行 smoke test：

```bash
cd /data/qty/trackerless-us-reconstruction
python -m baselines.nr_rec_fus.preprocessing.prepare_nr_rec_fus
python -m baselines.nr_rec_fus.train --steps 1 --device cpu
```

训练时将 `--device cuda` 并设置 `CUDA_VISIBLE_DEVICES=1`。训练输入只使用超声帧；训练中的 tracker 位姿用于刚性监督，推理阶段不输入位姿。当前实现采用 TUS-REC2024 官方 train/val 划分，不能把论文原始数据上的数值直接当作本数据集结果。
