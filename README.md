# 无定位三维自由手超声重建：任务理解

整理日期：2026-09-19。本文依据 **Trackerless 3D Freehand Ultrasound Reconstruction Challenge 2024（TUS-REC2024）** 的官方任务说明、数据说明、评估协议和提交接口整理，并对照现有 `../neural-ex/` 项目。以下挑战赛定义均针对 2024 年版本。

数据集下载、官方划分、校验方式和读取示例见 [DATASET.md](DATASET.md)。运行 `python scripts/dataset_status.py` 可查看准备进度。

Long-Term Dependency 方法的统一预处理和复现代码见[下方说明](#long-term-dependency-复现)。

## 1. 核心任务定义

给定按时间排序的二维超声图像序列，在推理时不使用外部跟踪器提供的逐帧探头位姿，从图像中恢复各帧的三维空间关系，将整段扫描重建到统一的三维坐标系中。

“无定位”指推理时不依赖外部跟踪器测得的逐帧位姿。训练数据仍可包含跟踪器位姿，作为监督真值；已知的像素尺度和探头标定信息也可以使用。因此，无定位重建不等同于无监督学习，也不意味着所有几何信息都未知。

## 2. 输入与输出

### 输入

官方 `predict_ddfs` 接口接收：

- `frames`：整段扫描的二维超声图像，形状为 `[N, 480, 640]`，其中 `N` 是帧数。
- `landmark`：20 个标志点的帧编号和二维图像坐标，形状为 `[20, 3]`。这些信息指定需要输出位移的位置，不提供三维位置真值。
- `data_path_calib`：标定文件路径，包含像素到毫米的尺度以及图像坐标系到跟踪工具坐标系的空间标定。

推理接口不提供逐帧真实位姿。

### 输出

比赛采用三维位移向量表示预测的空间关系，单位为毫米，要求四组输出：

| 输出 | 含义 | 参考帧 | 形状 |
| --- | --- | --- | --- |
| GP | 所有像素的全局位移 | 扫描的第一帧 | `[N-1, 3, 307200]` |
| GL | 标志点的全局位移 | 扫描的第一帧 | `[3, 20]` |
| LP | 所有像素的局部位移 | 当前帧的前一帧 | `[N-1, 3, 307200]` |
| LL | 标志点的局部位移 | 当前帧的前一帧 | `[3, 20]` |

其中 `307200 = 480 × 640`。具体坐标轴、像素展平顺序、标志点索引和位移方向应以官方转换代码为准。

比赛的目标是三维重建，但提交评估接口要求的是上述位移，而不是必须输出一个栅格化三维灰度体。

## 3. 方法可以采用哪些形式

比赛不强制指定算法内部结构，允许：

- 学习方法或非学习方法。
- 逐帧、帧对、序列或整段扫描处理。
- 刚性、仿射或非刚性空间变换。

预测每帧六自由度位姿是可行实现之一，并不是任务的强制限定。刚性方法可以先预测变换矩阵，再转换成比赛要求的四组位移。整段序列可用于推理，任务没有强制在线或仅使用历史帧。

第一帧作为全局参考，因此无需恢复扫描在外部世界坐标系中的绝对位置。

## 4. 局部关系、全局关系与累积漂移

局部预测描述当前帧相对于前一帧的空间关系，全局预测描述当前帧相对于第一帧的空间关系。

以刚性方法为例，令预测的相邻帧变换为 $\hat T_{i-1\leftarrow i}$，即把第 $i$ 帧的点映射到第 $i-1$ 帧的参考坐标系，则：

$$
\hat T_{0\leftarrow i}
=\hat T_{0\leftarrow1}\hat T_{1\leftarrow2}\cdots\hat T_{i-1\leftarrow i}.
$$

这里各矩阵必须采用一致的坐标约定；图像坐标与工具坐标之间的转换需结合标定处理。

即使每一步的误差很小，连乘后的长序列轨迹仍可能发生明显漂移。因此，相邻帧关系准确不代表整段扫描几何准确，需要同时评价局部和全局重建。

## 5. 评估：三维位置误差

官方评价预测重建位置与真值重建位置之间的欧氏距离：

$$
E=\frac{1}{|P|}\sum_{p\in P}\left\|\hat{\mathbf x}_{p}-\mathbf x_{p}^{GT}\right\|_2.
$$

在不同参考帧和点集上计算，得到四项指标：

| 指标 | 名称 | 评价内容 |
| --- | --- | --- |
| GPE | Global Pixel Reconstruction Error | 所有像素相对于第一帧的三维位置误差 |
| GLE | Global Landmark Reconstruction Error | 标志点相对于第一帧的三维位置误差 |
| LPE | Local Pixel Reconstruction Error | 所有像素相对于前一帧的三维位置误差 |
| LLE | Local Landmark Reconstruction Error | 标志点相对于前一帧的三维位置误差 |

四项原始误差均以毫米计，越小越好。局部指标反映相邻帧几何准确性，全局指标反映整段扫描的几何准确性，包括误差累积的影响。

官方排名对每个测试扫描的各项误差，依据参赛方法中的最优和最差结果归一化为分数，再以各 25% 的权重求和，最后在测试扫描间平均；最终分数越高越好。运行时间也会被记录，同分时用于排名。

**这里的 pixel reconstruction error 是像素在三维空间中的位置误差，不是像素灰度 MSE。** PSNR、SSIM 等图像指标可补充评价灰度重建质量，但不能替代上述几何指标，也不能单独证明空间重建准确。

## 6. 2024 年数据集与划分

根据 2024 年官方数据页面：

- 数据来自 85 名健康志愿者的左右前臂，共 2040 段扫描，每人 24 段。
- 按受试者划分：训练 50 人／1200 段，验证 3 人／72 段，测试 32 人／768 段。
- 图像大小为 `480 × 640`，采集帧率为 20 fps。
- 扫描协议包含直线、C 形和 S 形轨迹，远端到近端及反向扫描，以及超声平面与扫描方向平行或垂直的情况。
- 训练数据的 `frames` 形状为 `[N, H, W]`；`tforms` 形状为 `[N, 4, 4]`，表示跟踪工具坐标系到相机坐标系的变换。
- 提供像素尺度和空间标定矩阵，用于建立图像像素与物理空间之间的关系。

部分早期仓库背景介绍提及“一百名志愿者”，本文采用 2024 年官方数据页面明确给出的实际数量与划分。

按受试者划分意味着需要在未见过的受试者上验证泛化能力。同一扫描内部留出部分帧，不能直接替代该测试设置。

## 7. 与现有 neural-ex 有定位重建的关系

| 方面 | `neural-ex` 当前有定位主流程 | TUS-REC2024 无定位重建 |
| --- | --- | --- |
| 推理时逐帧位姿 | 已知，由位姿文件提供 | 不提供，需要从图像估计空间关系 |
| 核心学习目标 | 三维坐标到超声灰度的映射 | 图像序列到空间变换或位移的映射 |
| 主要评估 | 留出切片的灰度重建质量 | 局部、全局三维位置准确性 |
| 当前数据划分 | 同一扫描每五帧留出一帧 | 按受试者划分训练、验证和测试 |

`neural-ex` 的有定位主流程利用已知位姿，把二维像素经过尺度转换和标定后映射到三维空间，再学习连续灰度场：

$$
f_\theta(x,y,z)\rightarrow I.
$$

无定位场景增加了空间布局估计这一关键问题。一个与现有代码衔接的实现方向是：

```text
二维超声序列
    ↓
估计逐帧位姿或空间位移
    ↓
建立统一三维空间中的像素位置
    ↓
利用 INR／混合专家模型构建连续超声灰度场
```

这是一种可选技术路线，并非比赛强制要求；也可以研究联合优化空间布局与灰度表示的方法。

现有 `../neural-ex/bridge_nr_rec_fus/` 提供“位姿估计 → INR 重建 → 对比真实位姿与预测位姿”的流程，与上述路线衔接。但仅存在该流程不代表已完成挑战赛协议验证，仍需核对推理输入、坐标约定、独立受试者划分，并补充 GP、GL、LP、LL 对应的几何评估。

## 8. 本目录后续工作的任务表述

给定无逐帧定位信息的二维超声序列，估计其三维空间布局，并据此构建三维超声表示。通过空间位置误差评价几何准确性，通过图像指标补充评价灰度重建质量。训练可使用跟踪器真值监督，推理不使用逐帧真实位姿，并在独立受试者上验证泛化能力。

## Long-Term Dependency 复现

当前目录提供论文方法针对 TUS-REC2024 的可运行适配。预处理保持原始 HDF5 不变，只生成索引、4 倍下采样标定矩阵和配置：

```bash
python scripts/preprocess_longterm.py
```

训练设置与作者实现一致：输入连续 10 帧，采样范围为 10 帧，预测最后 9 帧相对前面帧的 45 个帧对变换；EfficientNet-B1 接收 10 通道灰度输入，输出每个帧对的 6-DoF 参数。监督将相对变换作用到四个图像角点，再在工具坐标系中计算点坐标 MSE。训练时使用跟踪器位姿作为监督，模型输入不包含位姿。

```bash
python train_longterm.py \
  --root data/tus-rec2024 \
  --preprocessed data/tus-rec2024/preprocessed/longterm \
  --output runs/longterm_v2 \
  --device cuda
```

默认训练 100 个 epoch、批大小 32、学习率 `1e-4`。没有 GPU 时可用 `--device cpu --max-steps 1 --val-steps 1 --epochs 1` 做流程检查；步数限制不代表论文结果。新训练的最佳验证模型保存在 `runs/longterm_v2/best.pt`，训练历史保存在 `runs/longterm_v2/history.json`。

这是在 TUS-REC2024 上的统一复现版本：官方 2024 数据按受试者划分为 000–049 训练、050–052 验证，而论文原始实验使用作者发布的旧数据和交叉验证。因此，不能把本实验结果直接当作论文表格的数值；比较时应在同一 TUS-REC2024 划分上重新训练所有方法。

## 官方来源

1. [TUS-REC2024 任务说明](https://github-pages.ucl.ac.uk/tus-rec-challenge/TUS-REC2024/task.html)
2. [TUS-REC2024 数据集说明](https://github-pages.ucl.ac.uk/tus-rec-challenge/TUS-REC2024/data.html)
3. [TUS-REC2024 评估协议](https://github-pages.ucl.ac.uk/tus-rec-challenge/TUS-REC2024/assessment.html)
4. [TUS-REC2024 提交接口](https://github-pages.ucl.ac.uk/tus-rec-challenge/TUS-REC2024/submission.html)
5. [官方基线代码](https://github.com/QiLi111/tus-rec-challenge_baseline)
6. [官方提交代码说明](https://github.com/QiLi111/tus-rec-challenge_baseline/blob/main/submission/README.md)
7. [位姿到位移的官方转换代码](https://github.com/QiLi111/tus-rec-challenge_baseline/blob/main/submission/utils/Transf2DDFs.py)

## 两个基线的统一训练与评估

两个方法现已共用受试者隔离检查、确定性窗口验证、完整训练状态保存和整段扫描评估。NR 的三维体配准适配及其与原论文的差异见 [NR-Rec-FUS 说明](baselines/nr_rec_fus/README.md)。该实现属于方法适配，不能声称精确复现原论文结果。

### 训练与恢复

```bash
# Long-Term：新训练默认写入 runs/longterm_v2，保留此前 runs/longterm
/data/qty/anaconda3/bin/python train_longterm.py --device cuda

# NR：默认写入 runs/nr_rec_fus
/data/qty/anaconda3/bin/python -m baselines.nr_rec_fus.train \
  --config baselines/nr_rec_fus/configs/tus_rec2024.json --device cuda
```

每个 epoch 都保存 `last.pt`、按固定窗口验证距离选择的 `best.pt`、`history.json`、`config.json`，并更新 `loss_curves.png` 和矢量版 `loss_curves.svg`。曲线包含总损失、点距离、刚性距离、细化损失和三维配准损失的 train/val 对照。检查点包含优化器、随机数状态和历史；恢复时要求采样、验证步数及批大小等设置一致。验证固定取每段扫描的前／中／后三个窗口，按窗口数加权平均；训练仍每个 epoch 为每段扫描取一个窗口。未指定 `--resume` 时禁止覆盖已有结果。CPU 必须显式指定 `--device cpu`。

```bash
/data/qty/anaconda3/bin/python train_longterm.py --device cuda \
  --resume runs/longterm_v2/last.pt

/data/qty/anaconda3/bin/python -m baselines.nr_rec_fus.train --device cuda \
  --resume runs/nr_rec_fus/last.pt
```

Long-Term 默认仍为 10 帧、45 对、B1、100 epoch、batch 32；帧数参数现在同时控制模型和标签，避免只修改数据而造成维度错误。NR 默认连续 4 帧、B0、50 epoch、batch 2；这些训练预算及输入差异会记录在配置中。正式比较应报告这些差异，或显式选择一致的训练预算，不能直接比较训练 loss。

Long-Term 原有 `runs/longterm/best.pt` 可直接交给新评估器。若恢复旧格式权重训练，使用新输出目录并将 `--epochs` 设置为大于检查点 epoch 的目标总数。旧检查点没有完整随机数状态，无法保证与旧脚本未中断训练完全一致；旧随机验证与新固定验证不可直接拼接，迁移时重新建立最佳验证记录。Long-Term 训练保留旧版零起点四角点约定与预处理标定，以兼容已有权重；完整扫描评估独立使用官方像素坐标。

### 四项几何指标

```bash
# 可直接评估已经训练好的 Long-Term
/data/qty/anaconda3/bin/python evaluate.py \
  --checkpoint runs/longterm/best.pt --device cuda \
  --output runs/eval_longterm

# 训练 NR 后用完全相同的验证扫描与评分器评估
/data/qty/anaconda3/bin/python evaluate.py \
  --checkpoint runs/nr_rec_fus/best.pt --device cuda \
  --output runs/eval_nr_rec_fus
```

默认评估全部 72 段验证扫描和原始 480×640 的全部像素；标志点按官方坐标直接读取。模型只接收图像，评分阶段才读取位姿。两个方法使用相同的窗口拼接规则：步长为输入帧数减一，以已经放置的窗口首帧为锚点，保留已有重叠帧，补齐尾窗口；短于输入长度的扫描重复末帧填充。Long-Term 选择每个窗口的 `(0,j)` 长距离预测，NR 选择同样的首帧相对预测。

输出包括：

- `metrics.json`：方法配置、扫描数、四项误差均值／标准差、推理时间，以及是否只评估部分扫描。
- `per_scan.jsonl`：逐扫描 GPE、GLE、LPE、LLE（毫米，越小越好）。最终均值对扫描等权，不能把长扫描当作更高权重。
- 每段扫描的 `prediction.npz`：预测全局工具变换，以及 NR 使用的三维形变场与体边界。

评分分块计算全部像素，`--chunk-size` 只控制内存，不做像素抽样。`--max-scans 1` 可检查一段完整扫描；此时 `partial=true`，不能充当完整验证集结果。`--rigid-only` 可评估 NR 的刚性分支消融。验证集用于选模型，结果不能替代官方隐藏测试集泛化结论；本工具也不计算依赖其他参赛方法结果的官方归一化排名分数。

### 坐标与提交接口

转换依据官方 `submission/utils/{plot_functions,transform,Transf2DDFs}.py`：原始像素 `x=1..640, y=1..480`，按行展平且 x 最快变化；相对工具变换通过 `C^-1 T C` 转入参考图像毫米坐标，再作用于 `S p`。标志点帧编号 `k` 对应原序列 `frames[k]`、不含首帧的变换数组第 `k-1` 项；有效范围为 `1..N-1`。不对标志点坐标再加减一。

`predict_ddfs.py` 提供官方形状的 `predict_ddfs(frames, landmark, data_path_calib)`，返回顺序是 GP、GL、LP、LL。设置 `TUSREC_CHECKPOINT` 指向任一方法权重，`TUSREC_DEVICE` 默认为 `cuda`。接口不接收真实位姿；完整挑战赛容器打包仍需按主办方环境完成。

直接返回两组全像素位移需要约 `7.37 MB × (N-1)` 内存。离线评估通常只需要四项误差，不必物化这些数组；若需要保存位移，给 `evaluate.py` 添加 `--export-ddfs`，使用 `.npy` 内存映射逐块写入 GP、LP，并保存 GL、LL。

### 流程检查

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/qty/anaconda3/bin/python \
  -m unittest discover -s tests -v

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/qty/anaconda3/bin/python \
  train_longterm.py --device cpu --epochs 1 --batch-size 2 --workers 0 \
  --max-steps 1 --val-steps 1 --output runs/longterm_smoke_v2

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/qty/anaconda3/bin/python \
  -m baselines.nr_rec_fus.train --device cpu --epochs 1 \
  --steps 1 --val-steps 1 --output runs/nr_smoke_v2
```

步数限制会记录为 `debug_run=true`，这些运行仅用于验证流程，不能解释为方法性能。

## MoGLo-Net 复现

MoGLo-Net（Lee et al., IEEE TMI 2025）公开实现使用共享图像编码器、patch-wise correlation、global-local self-attention、双 LSTM 运动头和 MME／相关性／triplet 复合损失。作者代码和原始数据协议见 [US3D](https://github.com/pnu-amilab/US3D)。本目录提供针对 TUS-REC2024 的适配：输入连续 5 帧，模型输入为 120×160 灰度图，监督为相邻帧的标定毫米坐标相对变换；训练推理均不输入跟踪器位姿。

作者论文实验使用 Forearm_Main 数据、特定 `y_scale`、256×256 图像、1001 epoch 和其自定义划分。本适配使用 TUS-REC2024 的 000–049／050–052 受试者划分、公开挑战赛的标定和 GP／GL／LP／LL 评估，因此不是论文表格的数值复现。配置中的 `motion_scale` 只用于稳定 6-DoF 回归，预测时会还原到毫米和弧度。

```bash
/data/qty/anaconda3/bin/python -m baselines.moglo_net.train \
  --config baselines/moglo_net/configs/tus_rec2024.json --device cuda
```

默认输出 `runs/moglo_net/{best.pt,last.pt,history.json,steps.jsonl,loss_curves.png,loss_curves.svg}`；最佳模型按固定的三个验证窗口选择。整段扫描评估与两个已有基线使用同一评分器：

```bash
/data/qty/anaconda3/bin/python evaluate.py \
  --checkpoint runs/moglo_net/best.pt --method moglo_net --device cuda \
  --output runs/eval_moglo_net
```

完整 GPU 实验脚本会保存环境、源文件哈希、命令日志、逐批次损失、逐扫描预测／真值／误差、损失曲线和校验清单：

```bash
/data/qty/anaconda3/bin/python scripts/run_moglo_experiment.py \
  --output runs/experiments/moglo_20260920_gpu0 --gpu 0
```

### 双 GPU 实验与原始记录

`scripts/run_baseline_experiment.py` 用物理 GPU 0 评估已经完成 100 epoch 的 `runs/longterm/best.pt`，用物理 GPU 1 按配置训练 NR，再评估 NR 完整模型和刚性分支。它保存代码快照、Git 版本及差异、环境版本、数据索引与标定、源文件大小和修改时间、权重校验值、所有子进程命令／退出码／日志及每 30 秒 GPU 状态。输出目录必须是新目录。

```bash
/data/qty/anaconda3/bin/python scripts/run_baseline_experiment.py --output runs/experiments/my_gpu01_run
```

查询已有实验的进度（不会操作 GPU 进程）：

```bash
/data/qty/anaconda3/bin/python scripts/experiment_status.py runs/experiments/20260919_gpu01
```

新训练额外保存 `steps.jsonl`：每个训练批次和每个验证窗口的全部损失分量、毫米距离、扫描编号、帧索引、学习率与时间。旧 Long-Term 只有原始 epoch 历史，无法补造历史逐批次数据；实验归档会从该历史重绘标准曲线。也可以手动运行 `scripts/plot_training_history.py runs/.../history.json`。

评估默认在 `--device` 指定的设备上计算几何指标；可用 `--metric-device cpu` 单独选择 CPU 评分。每段扫描保存：

- `prediction.npz`：完整预测变换、像素标定、空间标定及 NR 的形变场／体边界。
- `reference.npz`：对应真实位姿与标志点，仅用于评估和结果复核。
- `errors.npz`：逐帧全像素平均误差、20 个标志点各自的误差、预测及真实 GL／LL。

这些记录可重新生成全部 GP／GL／LP／LL 并重新评分，不需要重复存储原始超声图像或数百 GB 的全像素位移。实验结束后生成 `comparison.json`、`per_scan_comparison.csv`、`REPORT.md` 和产物 SHA-256 清单。报告明确保留两种方法训练预算和检查点选择协议不同这一限制。
