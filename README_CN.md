<br>
<p align="center">
<h1 align="center"><strong>EgoTraj-Bench：面向第一人称视角噪声观测的鲁棒轨迹预测</strong></h1>
  <p align="center">
    <a href='https://www.jiayi-liu.cn/' target='_blank'>Jiayi Liu</a><sup>1</sup>&emsp;
    <a href='https://jiaming-zhou.github.io/' target='_blank'>Jiaming Zhou</a><sup>1</sup>&emsp;
    <a href='https://yipko.com/about/' target='_blank'>Ke Ye</a><sup>1</sup>&emsp;
    <a href='https://kunyulin.github.io/' target='_blank'>Kun-Yu Lin</a><sup>2</sup>&emsp;
    <a href='https://allanwangliqian.com/' target='_blank'>Allan Wang</a><sup>3</sup>&emsp;
    <a href='https://junweiliang.me/' target='_blank'>Junwei Liang</a><sup>1,4</sup>&emsp;
    <br>
    <sup>1</sup>香港科技大学（广州）&emsp;<sup>2</sup>香港大学&emsp;<sup>3</sup>日本科学未来馆&emsp;<sup>4</sup>香港科技大学
  </p>
</p>

<div id="top" align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2510.00405-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2510.00405)
[![Paper](https://img.shields.io/badge/Paper-PDF-0065D3?logo=readthedocs&logoColor=white)](https://arxiv.org/pdf/2510.00405)
[![Code](https://img.shields.io/badge/Code-GitHub-181717?logo=github&logoColor=white)](https://github.com/zoeyliu1999/EgoTraj-Bench)
[![Project](https://img.shields.io/badge/Project-Website-4285F4?logo=googlechrome&logoColor=white)](https://zoeyliu1999.github.io/EgoTrajBench/)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-HuggingFace-FFD21E)](https://huggingface.co/datasets/ZoeyLIU1999/EgoTraj-Bench)
[![Video](https://img.shields.io/badge/Video-YouTube-FF0000?logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=zrlAEPs9PAc)

</div>

> 📖 English version: [README.md](README.md)


![EgoTraj Intuition](assets/intuition.gif)

\* <span style="color:cyan">**青色**</span> 表示遮挡引起的轨迹断裂；<span style="color:red">**红色**</span> 表示 ID 切换；<span style="color:limegreen">**绿色**</span> 表示第一人称视角带来的透视畸变。

\* **虚线**：源自第一人称视角（FPV）；**实线**：源自鸟瞰视角（BEV）。

## 目录

- [关于本项目](#-关于本项目)
- [最新动态](#-最新动态)
- [项目结构](#-项目结构)
- [数据集](#-数据集)
- [模型](#-模型)
- [快速开始](#-快速开始)
  - [环境安装](#环境安装)
  - [数据准备](#数据准备)
  - [模型训练](#模型训练)
  - [模型评估](#模型评估)
- [预训练权重](#-预训练权重)
- [TODO 列表](#-todo-列表)
- [引用](#-引用)
- [许可协议](#-许可协议)
- [致谢](#-致谢)

---

## 🏠 关于本项目

**EgoTraj-Bench** 是一个面向真实场景、基于第一人称视角（ego-centric）噪声观测的鲁棒轨迹预测基准。它将带噪的第一人称视觉历史与干净的鸟瞰视角未来轨迹进行对齐，并显式建模真实世界中常见的感知伪影，例如**遮挡（occlusion）**、**ID 切换（ID switch）** 和 **跟踪漂移（tracking drift）**。


![Benchmark Overview](assets/benchmark.png)


## 🚀 最新动态

- **[2026-07]** 在 [HuggingFace](https://huggingface.co/datasets/ZoeyLIU1999/EgoTraj-Bench/tree/main/L1-intermediate) 发布 **L1 中间层** 核心数据，包含 BEV 真值、FPV 检测/跟踪结果、可见性元数据、机器人运动轨迹，以及 [L1 使用指南](README_L1.md)。
- **[2026-04]** 在 [HuggingFace](https://huggingface.co/datasets/ZoeyLIU1999/EgoTraj-Bench) 发布基准数据集，包含 **L2 处理后** 数据和 **L0 原始数据** 获取说明。
- **[2026-04]** 发布 **BiFlow** 模型代码，包含训练/评估脚本、示例及权重下载流程。

---

## 📂 项目结构

```
EgoTraj-Bench/
├── cfg/
│   ├── biflow_k20.yml
│   └── biflow_t2fpv_k20.yml
├── data/
│   ├── _download_data.py
│   ├── egotraj/
│   └── t2fpv/
├── checkpoints/               # 下载的发布版权重（不纳入 git 版本管理）
│   └── <release_name>/
│       ├── config_updated.yml
│       └── models/
│           └── checkpoint_best.pt
├── loaders/
│   └── dataloader_egotraj.py
├── models/
│   ├── backbone_biflow.py
│   ├── flow_matching_biflow.py
│   ├── context_encoder/
│   │   └── tbd_encoder_score.py
│   ├── feature_fuser/
│   │   └── shared_fuser.py
│   ├── motion_decoder/
│   │   └── mtr_decoder_score.py
│   └── utils/
│       ├── common_layers.py
│       ├── contextual_scorer.py
│       └── polyline_encoder.py
├── trainer/
│   └── biflow_trainer.py
├── scripts/
│   ├── train_biflow.py
│   ├── eval_biflow.py
│   ├── run_train.sh
│   ├── run_eval.sh
│   └── run_eval_all.sh
├── utils/
│   ├── common.py
│   ├── config.py
│   ├── dataset_config.py
│   ├── normalization.py
│   └── utils.py
├── examples/
│   ├── README.md
│   ├── test_model.py
│   └── test_pipeline.py
├── requirements.txt
└── results/
    └── <cfg_name>/               # 例如 biflow_k20 或 biflow_t2fpv_k20
        └── <run_name>/
            ├── config_updated.yml
            ├── models/           # 训练权重保存于此
            ├── log/
            └── samples/
```

---


## 📦 数据集

EgoTraj-Bench 数据集托管于 HuggingFace：

**🤗 [ZoeyLIU1999/EgoTraj-Bench](https://huggingface.co/datasets/ZoeyLIU1999/EgoTraj-Bench)**

| 层级 | 目录 | 说明 | 大小 |
|-------|--------|-------------|------|
| **L2** | `L2-processed/` | 可直接用于训练与评估的 `.npz` 文件 | ~44 MB |
| **L1** | `L1-intermediate/` | 核心 CSV/TXT 中间数据：BEV 真值、FPV 检测/跟踪、可见性元数据、机器人运动轨迹 | ~302 MB |
| **L0** | `L0-raw/` | 指向 [TBD 原始数据集](https://kilthub.cmu.edu/authors/TBDLab_Admin/18437643) 的链接 | ~170 GB |

### L2-processed 数据格式

训练与评估使用 **L2-processed**（最小可复现）数据集。每个 `.npz` 文件包含 4 个数组：

| 键 | 形状 | 说明 |
|-----|-------|-------------|
| `all_obs` | `[N, 8, 7]` | 基于第一人称视角推导的带噪历史轨迹 |
| `all_pred` | `[N, 20, 7]` | 干净的 BEV 真值（8 帧过去 + 12 帧未来） |
| `num_peds` | `[S]` | 每个序列中的行人数量 |
| `seq_start_end` | `[S, 2]` | 序列在 agent 维度上的起止索引 |

每个时间步 7 个特征：`[x, y, orientation, img_x, img_y, valid_mask, agent_id]`

> **说明**：L2-processed 已足够用于完整的训练与评估复现。
> L1-intermediate 面向流水线分析、数据调试以及新的数据处理变体。

### L1 中间数据的用途

如果你希望在直接的 L2 训练/评估之外做更多工作，L1 发布数据非常有用：

- **轨迹预测分析**：对比 FPV 派生的带噪历史与干净的 BEV 真值，检查最终 L2 样本是如何生成的。
- **检测 / 跟踪分析**：研究 FPV bbox 跟踪、跟踪器 ID、ID 切换、置信度以及 BEV 投影噪声。
- **可见性 / 遮挡分析**：使用投影到 FPV 的 BEV 真值，配合 `px_count`、投影 bbox 和 IoU 元数据。
- **BEV-FPV 对齐**：利用帧 ID、时间戳与机器人路径，将中间行数据映射回原始 TBD 帧。

完整的 L1 数据说明及应用路径请见 [README_L1.md](README_L1.md)。

### 快速上手示例

```python
import numpy as np

data = np.load("L2-processed/EgoTraj-TBD/egotraj_tbd_test.npz")
noisy_history = data["all_obs"][:, :, :2]      # [N, 8, 2] FPV 带噪 xy 坐标
clean_past    = data["all_pred"][:, :8, :2]     # [N, 8, 2] BEV 干净的过去 xy 坐标
clean_future  = data["all_pred"][:, 8:, :2]     # [N, 12, 2] BEV 干净的未来 xy 坐标
valid_mask    = data["all_obs"][:, :, 5]         # [N, 8] 可见性掩码
```

---


## 🤖 模型

**BiFlow** 是我们提出的双流流匹配（flow matching）模型。它通过共享的隐空间表示，联合去噪带噪的第一人称历史轨迹并预测未来轨迹，并通过 **EgoAnchor** 机制增强对行人意图的鲁棒建模。

<p align="center">
  <img src="assets/model.png" alt="Model Overview" width="95%">
</p>

---


## 🛠️ 快速开始

### 环境安装

**官方可复现环境**：Python 3.9，CUDA

```bash
conda create -n biflow39 python=3.9 -y
conda activate biflow39
pip install -r requirements.txt
```

### 数据准备

通过以下命令下载并组织处理后的数据集（来源：HuggingFace）：

```bash
python data/_download_data.py
```

可选（自定义路径）：

```bash
python data/_download_data.py --data-dir ./data
```

### 模型训练

使用提供的 shell 脚本可以便捷地开始训练：

```bash
# EgoTraj-TBD
bash scripts/run_train.sh --fold_name tbd --gpu 0

# T2FPV-ETH（留一交叉验证，例如 eth fold）
bash scripts/run_train.sh --fold_name eth --gpu 0
```

脚本会根据 `--fold_name` 自动选择配置文件、数据来源与数据目录。你也可以直接调用训练脚本以获得更细粒度的控制：

```bash
python scripts/train_biflow.py \
    --cfg cfg/biflow_t2fpv_k20.yml \
    --fold_name eth \
    --data_source original_bal \
    --data_dir ./data/t2fpv \
    --epochs 150 \
    --gpu 0
```

**可选 fold 名称：**
| Fold | 配置文件 | 数据来源 | 说明 |
|------|--------|-------------|-------------|
| `tbd` | `biflow_k20.yml` | `original` | EgoTraj-TBD（真实第一人称数据） |
| `eth` | `biflow_t2fpv_k20.yml` | `original_bal` | T2FPV-ETH（留一验证，测试集为 ETH） |
| `hotel` | `biflow_t2fpv_k20.yml` | `original_bal` | T2FPV-ETH（留一验证，测试集为 Hotel） |
| `univ` | `biflow_t2fpv_k20.yml` | `original_bal` | T2FPV-ETH（留一验证，测试集为 Univ） |
| `zara1` | `biflow_t2fpv_k20.yml` | `original_bal` | T2FPV-ETH（留一验证，测试集为 Zara1） |
| `zara2` | `biflow_t2fpv_k20.yml` | `original_bal` | T2FPV-ETH（留一验证，测试集为 Zara2） |

权重与日志会保存在 `results/<experiment_tag>/` 目录下。

### 快速 Bash 工作流（训练 + 评估）

```bash
# 1) 训练（示例：TBD，输出到 results/）
bash scripts/run_train.sh --fold_name tbd --gpu 0

# 2) 评估训练输出（--run_name 会自动解析为 results/<cfg_name>/<run_name>）
bash scripts/run_eval.sh --fold_name tbd --run_name <run_name> --gpu 0
```

### 模型评估

支持两种评估入口：

```bash
# A）评估训练产出（results/）
# 单个 fold —— 使用 --run_name（自动解析为 results/<cfg_name>/<run_name>）
bash scripts/run_eval.sh --fold_name tbd  --run_name <run_name> --gpu 0
bash scripts/run_eval.sh --fold_name eth  --run_name <run_name> --gpu 0

# 或者通过 --ckpt_dir 显式指定路径
bash scripts/run_eval.sh --ckpt_dir results/biflow_k20/<run_name> --fold_name tbd --gpu 0

# 一次性评估所有 T2FPV-ETH folds（results 模式）
bash scripts/run_eval_all.sh --source results --ckpt_base results/biflow_t2fpv_k20 --gpu 0
```

```bash
# B）评估发布版权重（checkpoints/T2FPV-*）
# 单个 fold
bash scripts/run_eval.sh --fold_name eth --release_name T2FPV-eth --gpu 0

# 一次性评估所有 T2FPV-ETH folds（release 模式，默认）
bash scripts/run_eval_all.sh --source release --gpu 0
```

在 `results` 模式下，`run_eval_all.sh` 会从 `<ckpt_base>/<fold>/`（或为兼容性从 `<ckpt_base>/<fold>/ckpt/`）读取每个 fold。
在 `release` 模式下，则从 `checkpoints/T2FPV-<fold>/` 读取。

**评估指标**：ADE/FDE（min-of-K，K=1,3,5,20）、JADE/JFDE（联合指标）。

---


## 📥 预训练权重

预训练权重托管在 HuggingFace 上的 [ZoeyLIU1999/EgoTraj-Bench](https://huggingface.co/datasets/ZoeyLIU1999/EgoTraj-Bench/tree/main/models)。

| 模型 | 数据集 | Fold | 发布名 | ADE (K=20) | FDE (K=20) | 权重 |
|-------|---------|------|-------------|------------|------------|------------|
| BiFlow | EgoTraj-TBD | tbd | `EgoTraj-TBD` | 0.19 | 0.27 | [HF](https://huggingface.co/datasets/ZoeyLIU1999/EgoTraj-Bench/tree/main/models/EgoTraj-TBD) |
| BiFlow | T2FPV-ETH | eth | `T2FPV-eth` | 0.66 | 0.85 | [HF](https://huggingface.co/datasets/ZoeyLIU1999/EgoTraj-Bench/tree/main/models/T2FPV-eth) |
| BiFlow | T2FPV-ETH | hotel | `T2FPV-hotel` | 0.49 | 0.59 | [HF](https://huggingface.co/datasets/ZoeyLIU1999/EgoTraj-Bench/tree/main/models/T2FPV-hotel) |
| BiFlow | T2FPV-ETH | univ | `T2FPV-univ` | 0.91 | 1.08 | [HF](https://huggingface.co/datasets/ZoeyLIU1999/EgoTraj-Bench/tree/main/models/T2FPV-univ) |
| BiFlow | T2FPV-ETH | zara1 | `T2FPV-zara1` | 0.42 | 0.58 | [HF](https://huggingface.co/datasets/ZoeyLIU1999/EgoTraj-Bench/tree/main/models/T2FPV-zara1) |
| BiFlow | T2FPV-ETH | zara2 | `T2FPV-zara2` | 0.50 | 0.62 | [HF](https://huggingface.co/datasets/ZoeyLIU1999/EgoTraj-Bench/tree/main/models/T2FPV-zara2) |

### 下载方式

使用仓库根目录下提供的下载脚本：

```bash
# 下载全部权重至 checkpoints/
python data/_download_ckpt.py

# Dry-run 预览将要下载的文件
python data/_download_ckpt.py --dry-run
```

文件会直接放置在 `checkpoints/<release_name>/` 下，无需额外的复制步骤。

### 评估发布版权重

```bash
# 一次性评估所有 T2FPV folds
bash scripts/run_eval_all.sh --source release --gpu 0

# 单个 fold（示例）
bash scripts/run_eval.sh --fold_name zara2 --release_name T2FPV-zara2 --gpu 0
```

完整参数说明请参见 [模型评估](#模型评估)。

---

## 📝 TODO 列表

- [x] 发布基准数据集及下载说明。
- [x] 发布基准代码及仓库结构。
- [x] 发布预训练权重（T2FPV 各 folds 及 EgoTraj-TBD）。
- [x] 发布 L1 中间层核心数据及校验和。
- [x] 添加 L1 应用指南，支持下游分析。
- [x] 补充数据格式、评估指标与排行榜的详细文档。
- [x] 添加 EgoTraj-Bench 的使用示例与教程。

## 🔗 引用

如果我们的工作对您有帮助，欢迎 star 本仓库 🌟，并引用：

```bibtex
@inproceedings{liu2025egotraj,
    title   =   {EgoTraj-Bench: Towards Robust Trajectory Prediction Under Ego-view Noisy Observations},
    author  =   {Liu, Jiayi and Zhou, Jiaming and Ye, Ke and Lin, Kun-Yu and Wang, Allan and Liang, Junwei},
    booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
    year    =   {2025}
}
```

## 📄 许可协议

本项目采用 [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) 协议。

详情请见 [LICENSE](LICENSE) 文件。

## 👏 致谢

- [TBD Dataset](https://kilthub.cmu.edu/authors/TBDLab_Admin/18437643) —— 感谢其公开本基准所使用的原始第一人称轨迹数据。
- [T2FPV](https://github.com/cmubig/T2FPV) —— 感谢其在 ETH-UCY 上开源的仿真第一人称视角基准协议。
- [MoFlow](https://github.com/DSL-Lab/MoFlow) —— 感谢其开源的基线代码库，为本项目的实现与复现流水线提供了参考。
- 项目主页模板基于 [Nerfies](https://nerfies.github.io)。
