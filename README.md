<br>
<p align="center">
<h1 align="center"><strong>EgoTraj-Bench: Towards Robust Trajectory Prediction Under Ego-view Noisy Observations</strong></h1>
  <p align="center">
    <a href='https://www.jiayi-liu.cn/' target='_blank'>Jiayi Liu</a><sup>1</sup>&emsp;
    <a href='https://jiaming-zhou.github.io/' target='_blank'>Jiaming Zhou</a><sup>1</sup>&emsp;
    <a href='https://yipko.com/about/' target='_blank'>Ke Ye</a><sup>1</sup>&emsp;
    <a href='https://kunyulin.github.io/' target='_blank'>Kun-Yu Lin</a><sup>2</sup>&emsp;
    <a href='https://allanwangliqian.com/' target='_blank'>Allan Wang</a><sup>3</sup>&emsp;
    <a href='https://junweiliang.me/' target='_blank'>Junwei Liang</a><sup>1,4</sup>&emsp;
    <br>
    <sup>1</sup>HKUST(GZ)&emsp;<sup>2</sup>HKU&emsp;<sup>3</sup>Miraikan&emsp;<sup>4</sup>HKUST
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


![EgoTraj Intuition](assets/intuition.gif)

\* <span style="color:cyan">**Cyan**</span> highlights occlusion-induced gaps; <span style="color:red">**red**</span> indicates ID switches; <span style="color:limegreen">**green**</span> shows ego-centric perspective distortions.

\* **Dashed**: First person view derived; **Solid**: bird's eye view derived.

## Table of Contents

- [About](#-about)
- [News](#-news)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [L1 Intermediate Guide](README_L1.md)
- [Model](#-model)
- [Quick Start](#-quick-start)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Pretrained Checkpoints](#-pretrained-checkpoints)
- [TODO List](#-todo-list)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 🏠 About

**EgoTraj-Bench** is a real-world benchmark for robust trajectory prediction from ego-centric noisy observations. It grounds noisy first-person visual histories in clean bird's-eye-view future trajectories, explicitly modeling real-world perceptual artifacts such as occlusions, ID switches, and tracking drift.


![Benchmark Overview](assets/benchmark.png)


## 🚀 News

- **[2026-07]** **L1 intermediate** core data released on [HuggingFace](https://huggingface.co/datasets/ZoeyLIU1999/EgoTraj-Bench/tree/main/L1-intermediate), with BEV GT, FPV detections/tracks, visibility metadata, robot paths, and an [L1 application guide](README_L1.md).
- **[2026-04]** Benchmark dataset released on [HuggingFace](https://huggingface.co/datasets/ZoeyLIU1999/EgoTraj-Bench), including **L2 processed** data and **L0 raw-data** access instructions.
- **[2026-04]** Code for **BiFlow** model released, including training/evaluation scripts, examples, and checkpoint download workflow.

---

## 📂 Project Structure

```
EgoTraj-Bench/
├── cfg/
│   ├── biflow_k20.yml
│   └── biflow_t2fpv_k20.yml
├── data/
│   ├── _download_data.py
│   ├── egotraj/
│   └── t2fpv/
├── checkpoints/               # downloaded release checkpoints (not tracked by git)
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
    └── <cfg_name>/               # e.g., biflow_k20 or biflow_t2fpv_k20
        └── <run_name>/
            ├── config_updated.yml
            ├── models/           # checkpoints saved here
            ├── log/
            └── samples/
```

---


## 📦 Dataset

The EgoTraj-Bench dataset is available on HuggingFace:

**🤗 [ZoeyLIU1999/EgoTraj-Bench](https://huggingface.co/datasets/ZoeyLIU1999/EgoTraj-Bench)**

| Level | Folder | Description | Size |
|-------|--------|-------------|------|
| **L2** | `L2-processed/` | Ready-to-use `.npz` files for training and evaluation | ~44 MB |
| **L1** | `L1-intermediate/` | Core CSV/TXT intermediates: BEV GT, FPV detections/tracks, visibility metadata, robot paths | ~302 MB |
| **L0** | `L0-raw/` | Link to [TBD raw dataset](https://kilthub.cmu.edu/authors/TBDLab_Admin/18437643) | ~170 GB |

### L2-processed Data Format

Training and evaluation use the **L2-processed** (minimal reproducible) set.
Each `.npz` file contains 4 arrays:

| Key | Shape | Description |
|-----|-------|-------------|
| `all_obs` | `[N, 8, 7]` | Noisy FPV-derived past trajectories |
| `all_pred` | `[N, 20, 7]` | Clean BEV ground truth (8 past + 12 future) |
| `num_peds` | `[S]` | Number of agents per sequence |
| `seq_start_end` | `[S, 2]` | Sequence index boundaries in the agent dimension |

7 features per timestep: `[x, y, orientation, img_x, img_y, valid_mask, agent_id]`

> **Note**: L2-processed is sufficient for full training/evaluation reproduction.
> L1-intermediate is released for pipeline analysis, data debugging, and new data-processing variants.

### L1 Intermediate Applications

The L1 release is useful when you want to go beyond direct L2 training/evaluation:

- **Trajectory prediction analysis**: compare noisy FPV-derived histories with clean BEV GT and inspect how final L2 samples are formed.
- **Detection / tracking analysis**: study FPV bbox tracks, tracker IDs, ID switches, confidence, and BEV projection noise.
- **Visibility / occlusion analysis**: use BEV GT projected into FPV with `px_count`, projected bboxes, and IoU metadata.
- **BEV-FPV alignment**: map intermediate rows back to raw TBD frames using frame IDs, timestamps, and robot paths.

See [README_L1.md](README_L1.md) for the full L1 data guide and application tracks.

### Quick Example

```python
import numpy as np

data = np.load("L2-processed/EgoTraj-TBD/egotraj_tbd_test.npz")
noisy_history = data["all_obs"][:, :, :2]      # [N, 8, 2] FPV noisy xy
clean_past    = data["all_pred"][:, :8, :2]     # [N, 8, 2] BEV clean past xy
clean_future  = data["all_pred"][:, 8:, :2]     # [N, 12, 2] BEV clean future xy
valid_mask    = data["all_obs"][:, :, 5]         # [N, 8] visibility mask
```

---


## 🤖 Model

**BiFlow**, our dual-stream flow matching model, jointly denoises noisy ego-centric histories and predicts future trajectories via a shared latent representation, enhanced by an EgoAnchor mechanism for robust intent modeling.

<p align="center">
  <img src="assets/model.png" alt="Model Overview" width="95%">
</p>

---


## 🛠️ Quick Start

### Installation

**Official reproducibility environment**: Python 3.9, CUDA

```bash
conda create -n biflow39 python=3.9 -y
conda activate biflow39
pip install -r requirements.txt
```

### Data Preparation

Download and organize the processed dataset (HuggingFace source) with:

```bash
python data/_download_data.py
```

Optional (custom paths):

```bash
python data/_download_data.py --data-dir ./data
```

### Training

Use the provided shell scripts for convenient training:

```bash
# EgoTraj-TBD
bash scripts/run_train.sh --fold_name tbd --gpu 0

# T2FPV-ETH (leave-one-out cross-validation, e.g., eth fold)
bash scripts/run_train.sh --fold_name eth --gpu 0
```

The script automatically selects the config, data source, and data directory based on `--fold_name`. You can also call the training script directly for more control:

```bash
python scripts/train_biflow.py \
    --cfg cfg/biflow_t2fpv_k20.yml \
    --fold_name eth \
    --data_source original_bal \
    --data_dir ./data/t2fpv \
    --epochs 150 \
    --gpu 0
```

**Available fold names:**
| Fold | Config | Data Source | Description |
|------|--------|-------------|-------------|
| `tbd` | `biflow_k20.yml` | `original` | EgoTraj-TBD (real-world ego-centric) |
| `eth` | `biflow_t2fpv_k20.yml` | `original_bal` | T2FPV-ETH (leave-one-out, test on ETH) |
| `hotel` | `biflow_t2fpv_k20.yml` | `original_bal` | T2FPV-ETH (leave-one-out, test on Hotel) |
| `univ` | `biflow_t2fpv_k20.yml` | `original_bal` | T2FPV-ETH (leave-one-out, test on Univ) |
| `zara1` | `biflow_t2fpv_k20.yml` | `original_bal` | T2FPV-ETH (leave-one-out, test on Zara1) |
| `zara2` | `biflow_t2fpv_k20.yml` | `original_bal` | T2FPV-ETH (leave-one-out, test on Zara2) |

Checkpoints and logs are saved to `results/<experiment_tag>/`.

### Quick Bash Workflow (Train + Eval)

```bash
# 1) Train (example: TBD, output goes to results/)
bash scripts/run_train.sh --fold_name tbd --gpu 0

# 2) Evaluate training output (--run_name -> results/<cfg_name>/<run_name>)
bash scripts/run_eval.sh --fold_name tbd --run_name <run_name> --gpu 0
```

### Evaluation

Two evaluation entry points are supported:

```bash
# A) Training outputs (results/)
# Single fold — use --run_name (auto-resolves to results/<cfg_name>/<run_name>)
bash scripts/run_eval.sh --fold_name tbd  --run_name <run_name> --gpu 0
bash scripts/run_eval.sh --fold_name eth  --run_name <run_name> --gpu 0

# Or pass an explicit path with --ckpt_dir
bash scripts/run_eval.sh --ckpt_dir results/biflow_k20/<run_name> --fold_name tbd --gpu 0

# All T2FPV-ETH folds at once (results mode)
bash scripts/run_eval_all.sh --source results --ckpt_base results/biflow_t2fpv_k20 --gpu 0
```

```bash
# B) Release checkpoints (checkpoints/T2FPV-*)
# Single fold
bash scripts/run_eval.sh --fold_name eth --release_name T2FPV-eth --gpu 0

# All T2FPV-ETH folds at once (release mode, default)
bash scripts/run_eval_all.sh --source release --gpu 0
```

For `run_eval_all.sh` in `results` mode, each fold is read from `<ckpt_base>/<fold>/` (or `<ckpt_base>/<fold>/ckpt/` for compatibility).
In `release` mode, it reads from `checkpoints/T2FPV-<fold>/`.

**Metrics reported**: ADE/FDE (min-of-K, K=1,3,5,20), JADE/JFDE (joint metrics).

---


## 📥 Pretrained Checkpoints

Checkpoints are hosted on HuggingFace under
[ZoeyLIU1999/EgoTraj-Bench](https://huggingface.co/datasets/ZoeyLIU1999/EgoTraj-Bench/tree/main/models).

| Model | Dataset | Fold | Release Name | ADE (K=20) | FDE (K=20) | Checkpoint |
|-------|---------|------|-------------|------------|------------|------------|
| BiFlow | EgoTraj-TBD | tbd | `EgoTraj-TBD` | 0.19 | 0.27 | [HF](https://huggingface.co/datasets/ZoeyLIU1999/EgoTraj-Bench/tree/main/models/EgoTraj-TBD) |
| BiFlow | T2FPV-ETH | eth | `T2FPV-eth` | 0.66 | 0.85 | [HF](https://huggingface.co/datasets/ZoeyLIU1999/EgoTraj-Bench/tree/main/models/T2FPV-eth) |
| BiFlow | T2FPV-ETH | hotel | `T2FPV-hotel` | 0.49 | 0.59 | [HF](https://huggingface.co/datasets/ZoeyLIU1999/EgoTraj-Bench/tree/main/models/T2FPV-hotel) |
| BiFlow | T2FPV-ETH | univ | `T2FPV-univ` | 0.91 | 1.08 | [HF](https://huggingface.co/datasets/ZoeyLIU1999/EgoTraj-Bench/tree/main/models/T2FPV-univ) |
| BiFlow | T2FPV-ETH | zara1 | `T2FPV-zara1` | 0.42 | 0.58 | [HF](https://huggingface.co/datasets/ZoeyLIU1999/EgoTraj-Bench/tree/main/models/T2FPV-zara1) |
| BiFlow | T2FPV-ETH | zara2 | `T2FPV-zara2` | 0.50 | 0.62 | [HF](https://huggingface.co/datasets/ZoeyLIU1999/EgoTraj-Bench/tree/main/models/T2FPV-zara2) |

### Downloading

Use the provided download script (from the repository root):

```bash
# Download all checkpoints to checkpoints/
python data/_download_ckpt.py

# Dry-run to preview which files will be fetched
python data/_download_ckpt.py --dry-run
```

Files are placed directly under `checkpoints/<release_name>/` — no intermediate copy step needed.

### Evaluating Release Checkpoints

```bash
# All T2FPV folds at once
bash scripts/run_eval_all.sh --source release --gpu 0

# Single fold (example)
bash scripts/run_eval.sh --fold_name zara2 --release_name T2FPV-zara2 --gpu 0
```

See [Evaluation](#evaluation) for the full flag reference.

---

## 📝 TODO List

- [x] Release benchmark dataset and download instructions.
- [x] Release benchmark code and repository structure.
- [x] Release pretrained checkpoints (T2FPV folds and EgoTraj-TBD).
- [x] Release L1 intermediate core data and checksums.
- [x] Add L1 application guide for downstream analysis.
- [x] Add detailed documentation for data format, metrics, and leaderboard.
- [x] Add examples and tutorials for using EgoTraj-Bench.

## 🔗 Citation

If you find our work helpful, please consider starring this repo 🌟 and cite:

```bibtex
@inproceedings{liu2025egotraj,
    title   =   {EgoTraj-Bench: Towards Robust Trajectory Prediction Under Ego-view Noisy Observations},
    author  =   {Liu, Jiayi and Zhou, Jiaming and Ye, Ke and Lin, Kun-Yu and Wang, Allan and Liang, Junwei},
    booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
    year    =   {2025}
}
```

## 📄 License

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

See the [LICENSE](LICENSE) file for details.

## 👏 Acknowledgements

- [TBD Dataset](https://kilthub.cmu.edu/authors/TBDLab_Admin/18437643) — for releasing the raw ego-centric trajectory data used in this benchmark.
- [T2FPV](https://github.com/cmubig/T2FPV) — for releasing the simulated ego-view benchmark protocol on ETH-UCY.
- [MoFlow](https://github.com/DSL-Lab/MoFlow) — for open-sourcing the baseline codebase that informed our implementation and reproduction pipeline.
- Project page template is based on [Nerfies](https://nerfies.github.io).
