# RecON：TUS-REC2024 适配

本目录依据作者公开的 [Lmy0217/RecON](https://github.com/Lmy0217/RecON)（核对提交 `7de31140d894d0af3f7d8e31aa63b9269ddbcb61`）实现 **RecON: Online Learning for Sensorless Freehand 3D Ultrasound Reconstruction**（Medical Image Analysis 87, 2023，DOI `10.1016/j.media.2023.102810`）。训练和推理都不把当前扫描的跟踪器位姿输入模型；跟踪器位姿仅用于离线监督、验证选模和推理结束后的评分。

## 方法流程

公开方法包含三个阶段：

1. 位姿骨干：相邻两帧、两幅边缘图和两通道光流组成6通道输入，ResNet-18提取特征，空间ConvLSTM建模序列，回归相邻帧的平移和XYZ Euler角。监督包含平移L1、旋转L1、相关性损失和motion-weighted loss。
2. 全局形状判别器：将真实和预测轨迹分别重建为三维体，用3D ResNet-10学习全局解剖形状先验。
3. 在线学习：每条待测扫描都从相同离线权重重新开始，执行PSC（相关运动伪监督）、GAS（全局对抗形状）和FCC（留出帧内容一致性）优化，默认30轮。在线阶段结束后输出相邻帧变换并累积为整段轨迹。

最终检查点内嵌由000–049训练受试者全部1200段扫描各取一个确定性窗口生成的压缩PSC回放库（图像、训练标签及来源帧索引）。因此 `predict_ddfs` 在线适配时不会读取当前验证／测试扫描的位姿，也不会在运行时打开训练标签文件。检查点中的 `online_bank.metadata` 可审计其来源；减小 `--pseudo-bank-size` 只允许用于存储／流程调试。

## TUS-REC2024 适配与论文差异

- 使用官方000–049／050–052受试者划分和GP／GL／LP／LL毫米评估，而不是论文的Spine、Fetus、DDH数据与自定义划分。
- 默认训练窗口为从32帧范围采样的16帧，输入为120×160；论文公开配置使用300×300图像和按数据集／帧率变化的整段序列。
- 保留ImageNet预训练ResNet-18、ConvLSTM、3D ResNet-10、初始学习率 `1e-4` 且每30 epoch减半、200 epoch骨干、50 epoch判别器及30轮在线更新。ImageNet权重缓存到 `data/torch-cache`。
- 边缘预处理复现公开实现的Gaussian–Sobel–阈值流程。公开代码要求带NVIDIA Optical Flow的CUDA OpenCV；本适配使用确定性的CPU Farnebäck光流，以便标准OpenCV环境运行。
- GAS/FCC使用当前项目已有的可微三线性散点体重建和固定 `32×64×64` 网格，取代作者动态体素范围上的逆距离插值。这保证批处理和显存上界，但属于明确的方法适配。

因此本实现用于在同一TUS-REC2024协议下比较核心方法，不能称为论文表格数值的逐项复现。

## 两阶段训练

先训练骨干：

```bash
/data/qty/anaconda3/bin/python -m baselines.recon.train \
  --stage backbone --config baselines/recon/configs/tus_rec2024.json \
  --device cuda
```

再训练判别器并写入PSC回放库：

```bash
/data/qty/anaconda3/bin/python -m baselines.recon.train \
  --stage discriminator --config baselines/recon/configs/tus_rec2024.json \
  --backbone-checkpoint runs/recon_backbone/best.pt --device cuda
```

骨干默认输出到 `runs/recon_backbone`，判别器输出到 `runs/recon`。两个阶段均保存 `best.pt`、`last.pt`、`history.json`、`steps.jsonl`、配置和损失曲线，并支持 `--resume`。骨干按固定验证窗口的毫米距离选模；判别器按验证BCE选模。已有目录不会被静默覆盖。

## 整段扫描评估

```bash
/data/qty/anaconda3/bin/python evaluate.py \
  --checkpoint runs/recon/best.pt --method recon --device cuda \
  --output runs/eval_recon
```

正式协议使用检查点记录的30轮在线学习。`--online-epochs 0` 或较小数值仅供消融和流程检查，不能作为正式RecON结果。评估仍覆盖72段验证扫描和480×640全部像素，输出文件与其他方法一致。

完整实验归档可由单GPU监督脚本执行；输出目录必须不存在：

```bash
/data/qty/anaconda3/bin/python scripts/run_recon_experiment.py \
  --output runs/experiments/recon_full --gpu 0
```

脚本会保存环境、代码与数据元信息哈希、两阶段逐批次日志、GPU遥测、内嵌PSC来源、72段逐扫描原始记录、汇总报告和产物校验值。

CPU流程检查：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/qty/anaconda3/bin/python \
  -m baselines.recon.train --stage backbone --device cpu \
  --no-imagenet-pretrained --epochs 1 --steps 1 --val-steps 1 \
  --output runs/recon_backbone_smoke
```

任何步数限制、关闭ImageNet预训练或减少在线轮数的运行都属于debug／ablation，不代表方法性能。
