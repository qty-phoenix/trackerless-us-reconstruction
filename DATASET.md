# TUS-REC2024 数据集使用说明

本目录按官方受试者划分构建数据集，保留原始 `480×640` 灰度帧和毫米尺度。数据是否已经准备完成，以 `data/tus-rec2024/status.json` 为准；下载中的文件不视为可用训练数据。

## 官方来源与划分

| 划分／文件 | 受试者 | 扫描数 | 官方来源 |
| --- | --- | --- | --- |
| 训练 Part 1 | 000–024 | 600 | [Zenodo 11178509](https://zenodo.org/records/11178509) |
| 训练 Part 2 | 025–049 | 600 | [Zenodo 11180795](https://zenodo.org/records/11180795) |
| 训练标志点 | 000–049 | 1200 段对应的标志点 | [Zenodo 11355500](https://zenodo.org/records/11355500) |
| 验证集 | 050–052 | 72 | [Zenodo 12979481](https://zenodo.org/records/12979481) |
| 官方测试集 | 32 名独立受试者 | 768 | 由组织方持有，此目录不伪造或用验证集替代测试集 |

训练集两部分压缩文件共约 83.76 GB，解压约 186.39 GB；验证集解压约 10.68 GB。需要同时保留训练压缩包和解压文件时，预留约 285 GB 空间。官方文件元数据、下载地址和 MD5 保存于 `metadata/zenodo_*.json`。

## 目录

```text
data/tus-rec2024/
├── archives/                  # 训练原始 ZIP、下载中的 .part、训练 landmark.zip
│   └── val_overrides/          # 从官方 ZIP 补取的验证标志点
├── metadata/                  # 官方 Zenodo 元数据及验证 ZIP 成员清单
├── calib_matrix.csv
├── train/
│   ├── frames_transfs/000/…    # 每个 HDF5 包含 frames 和 tforms
│   └── landmarks/             # landmark_000.h5 … landmark_049.h5
├── val/
│   ├── frames/050/…           # 图像
│   ├── transfs/050/…          # 真值位姿，与图像分开
│   ├── landmark/              # 官方命名为单数 landmark
│   ├── calib_matrix.csv
│   └── dataset_keys.h5
├── manifests/
│   ├── train.jsonl            # 在数据通过校验后生成
│   └── val.jsonl
├── status.json
└── prepare.log
```

每条索引包含受试者、扫描名称、帧数、图像尺寸、位姿及标志点路径。所有路径相对于 `data/tus-rec2024`，不依赖代码中的绝对路径。每个受试者的扫描不跨划分。

## 当前机器上的复用与校验

训练集重新从官方地址下载。训练 ZIP、训练标志点 ZIP 和标定文件使用官方 MD5 校验。

本地已有 `/data/qty/tus_rec2024/data/Freehand_US_data_val.zip`，但其大小及内容与当前官方压缩包存在差异。对照远端 ZIP 中央目录后发现 `landmark_051.h5`、`landmark_052.h5` 的内容 CRC 不同，已从官方 ZIP 的对应字节范围补取这两份文件。解压过程中，每个文件均计算解压后 CRC32 和字节数，与官方 ZIP 成员清单比较。这是**逐成员内容校验**，不宣称本地旧压缩包通过了官方整包 MD5。其余原始文件不修改。

建索引还会检查图像／位姿帧数一致性、图像大小、变换矩阵形状与数值、标志点键，以及每个受试者是否完整包含 24 段扫描。压缩成员会完整读取以校验 CRC，图像数值检查抽查首、中、末帧。

## 查看进度与恢复

从项目目录执行：

```bash
python scripts/dataset_status.py
tail -f data/tus-rec2024/prepare.log
```

下载及准备进程运行时不要重复启动。若中断，可恢复下载：

本机使用独立后台监督进程管理下载和准备，日志为 `background.log` 和 `prepare.log`。在普通终端中，可用以下命令恢复整套后台任务（使用同一个 Python 环境）：

```bash
/data/qty/anaconda3/envs/FUS/bin/python scripts/background_dataset.py
```

后台锁避免重复启动。若日志报告网络重试耗尽或校验失败，先检查原因再恢复；`status.json` 中的下载等待状态本身不保证下载进程仍在运行，应同时检查后台日志。也可以分别在前台恢复下载和准备：

```bash
python scripts/download_training.py
```

此脚本并行下载两个训练 ZIP，支持断点续传。准备脚本可以同时在另一终端运行，等待下载完成后自动核对 MD5、解压、建索引：

```bash
/data/qty/anaconda3/envs/FUS/bin/python -u scripts/prepare_dataset.py \
  --local-val /data/qty/tus_rec2024/data/Freehand_US_data_val.zip --wait
```

本机现有 `FUS` 环境已包含 `numpy`、`h5py`；迁移到其他环境时安装 `requirements-data.txt`。验证阶段使用的旧压缩包是准备阶段的输入，解压后日常读取不再依赖该文件。

## 读取示例

```python
from datasets import TUSREC2024

root = 'data/tus-rec2024'

# 整段验证扫描：默认不读取或返回跟踪器真值。
val = TUSREC2024(root, split='val')
sample = val[0]
frames = sample['frames']             # uint8 [N,480,640]
landmarks = sample['landmarks']       # 原始官方帧号和二维坐标
calibration = sample['calibration']

# 训练窗口：长度 30，滑动步长 5，只在同一扫描内部取窗口。
train = TUSREC2024(root, split='train', window=30, stride=5, include_targets=True)
sample = train[0]
tool_to_world = sample['targets']['tool_to_world']  # float64 [30,4,4]
```

读取器可用于 PyTorch `DataLoader`。窗口训练可批处理；整段扫描长度不同，默认建议 `batch_size=1`。短于窗口的扫描不会生成窗口；末尾不足整步长时补一个重叠窗口覆盖最后一帧。

读取器不进行裁剪、缩放、坐标归一化或位姿参数化，避免预先绑定某个对比方法。`frames` 保留 uint8，模型输入可自行除以 255。标志点保持官方索引约定，不做未经验证的零基／一基转换；训练窗口不返回全扫描标志点。

`tool_to_world` 对应官方跟踪工具到相机坐标系的 `tforms`。以列向量表示，原始像素坐标映射为：

```text
p_world = T_frame @ C @ S @ [u,v,0,1]
```

其中 `S=calibration['pixel_to_mm']`，`C=calibration['image_mm_to_tool']`。生成比赛 DDF 时，仍需采用官方参考帧坐标系、变换方向和像素展平顺序，不能直接把世界坐标当作位移。

## 数据使用条件

按 [TUS-REC2024 官方政策](https://github-pages.ucl.ac.uk/tus-rec-challenge/TUS-REC2024/policies.html)，公开数据用于研究，禁止商业使用。发表研究时按官方要求引用数据集及相关论文。数据不纳入 Git。
