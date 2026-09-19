# NR-Rec-FUS：TUS-REC2024 适配

本目录是 [Nonrigid Reconstruction of Freehand Ultrasound Without a Tracker](https://arxiv.org/abs/2407.05767) 的**独立适配**，不是作者代码的逐项复刻。参考[作者实现](https://github.com/QiLi111/NR-Rec-FUS)的端到端 `rec_reg`、单通道三维配准思路：预测刚性位姿，生成三维超声体，再由仅输入预测体的网络学习形变。跟踪器真值仅用于训练损失与评分。

旧版本的独立二维帧间光流已替换为与位姿耦合的三维体配准。该结构变化不兼容旧 NR 模型权重，需要重新训练。

## 训练

从项目根目录执行：

```bash
/data/qty/anaconda3/bin/python -m baselines.nr_rec_fus.train \
  --config baselines/nr_rec_fus/configs/tus_rec2024.json \
  --device cuda
```

可用 `CUDA_VISIBLE_DEVICES=1` 指定物理 GPU 1；`--device cuda` 使用可见 GPU 中的默认设备。CUDA 不可用时会明确报错，CPU 检查需要显式指定 `--device cpu`。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/qty/anaconda3/bin/python \
  -m baselines.nr_rec_fus.train --device cpu \
  --epochs 1 --steps 1 --val-steps 1 --output runs/nr_smoke
```

`--steps`／`--max-steps` 限制每个 epoch 的训练批数，不跳过验证与保存；`--val-steps` 限制验证批数。任何非零限制都在配置中标记为 `debug_run`，不能用作正式实验结果。默认输出 `runs/nr_rec_fus/{best.pt,last.pt,history.json,config.json}`。已有结果的目录不会被新训练覆盖。

恢复训练：

```bash
/data/qty/anaconda3/bin/python -m baselines.nr_rec_fus.train \
  --device cuda --resume runs/nr_rec_fus/last.pt
```

`--epochs` 是目标总 epoch 数。检查点保存模型、优化器、随机数状态、完整历史和最佳验证指标。恢复时检查模型、损失、采样与验证配置是否匹配。

## 模型与监督

- 默认输入连续 4 帧、120×160，EfficientNet-B0 输出其余 3 帧相对窗口首帧的工具坐标变换。`model` 支持 `efficientnet_b0` 和 `efficientnet_b1`。
- 标定将像素映射到毫米坐标，刚性监督使用全部降采样像素的三维位置 MSE。
- 可微三线性散点累积生成三维体；体边界只取决于预测点和固定毫米边距，训练时也不使用真值来选择边界。
- 单输入三维 U-Net 从预测体生成毫米形变场；真值体仅用作配准损失目标。体配准的梯度经过重建体传回刚性网络。
- 总损失为 `rigid_point_MSE + refined_loss_weight × refined_point_MSE + reg_loss_weight × (volume_MSE + smooth_weight × bending_energy)`。默认权重分别为 1、1000、0.01，全部参与计算。
- 预测体和真值体的占据区域联合掩码用于体 MSE；弯曲能使用物理网格间距计算纯二阶及混合二阶导数。无真值非刚性位移标签，细化点监督仍来自跟踪器几何。
- 为固定第一帧，对三维点使用 `x_i' = G_i p + d(G_i p) - d(p)`。这里 `G_i` 为图像毫米坐标变换，`p` 是该像素在参考平面上的毫米位置。

与作者实现的差异：默认帧数／骨干是本项目的资源配置；使用紧凑三维 U-Net、32×64×64 自适应边界网格、最大 5 mm 的直接位移场、首帧固定与附加细化点监督。没有复刻作者的 100 帧默认设置、PCA 坐标优化、1 mm 固定间距网格、速度场积分或元学习。该版本不宣称形变可逆／微分同胚，不能直接对照论文数值。

## 验证与整段扫描比较

共用官方训练 1200 段、验证 72 段索引。验证集独立位姿文件通过 `tforms_path` 读取；每段扫描固定取前、中、后三个窗口。`val_distance_mm` 是细化后窗口点距离，`val_rigid_distance_mm` 同时记录细化前距离。最佳模型按前者选择；它们不是官方全扫描指标。

完整扫描使用固定窗口拼接刚性变换，再以整段预测几何生成三维体并细化。训练窗口与完整扫描的空间跨度不同，固定体尺寸对应的物理分辨率也不同，这是本适配需要在实验中检验的限制。

```bash
/data/qty/anaconda3/bin/python evaluate.py \
  --checkpoint runs/nr_rec_fus/best.pt --device cuda \
  --output runs/eval_nr_rec_fus

# 刚性分支消融，使用同一个检查点
/data/qty/anaconda3/bin/python evaluate.py \
  --checkpoint runs/nr_rec_fus/best.pt --device cuda --rigid-only \
  --output runs/eval_nr_rec_fus_rigid
```

全扫描输出 GPE、GLE、LPE、LLE（毫米），按扫描等权汇总。NR 的局部位移定义为相邻两帧细化点之差，再转入前一帧的刚性图像坐标基；这明确约定了非刚性情况下的参考系，零形变时与官方刚性转换一致。它没有求非刚性形变的逆映射。

公平比较命令、官方接口、输出文件和坐标约定见根目录 [README.md](../../README.md#两个基线的统一训练与评估)。
