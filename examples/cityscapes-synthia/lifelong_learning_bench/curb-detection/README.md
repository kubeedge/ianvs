# Curb Detection with Lifelong Learning on Cityscapes-SYNTHIA

This benchmark evaluates **lifelong learning** algorithms for autonomous driving curb detection. The model trains on a mix of real (Cityscapes) and simulated (SYNTHIA) data across multiple incremental rounds, measuring how well it transfers knowledge from simulation to real-world scenes.

---

## Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Step 1 — Ianvs Installation](#step-1--ianvs-installation)
4. [Step 2 — Dataset Preparation](#step-2--dataset-preparation)
5. [Step 3 — Configuration](#step-3--configuration)
6. [Step 4 — Run the Benchmark](#step-4--run-the-benchmark)
7. [Step 5 — Reading Results](#step-5--reading-results)
8. [File Structure](#file-structure)
9. [Troubleshooting](#troubleshooting)

---

## Overview

| Property | Value |
|---|---|
| Paradigm | Lifelong Learning |
| Algorithm | RFNet (RGB + Depth Fusion Network) |
| Dataset | Cityscapes (real) + SYNTHIA (simulated) |
| Task | Semantic segmentation — curb class detection |
| Metrics | `accuracy` (mIoU-based), `samples_transfer_ratio` |
| Incremental Rounds | 2 |

**Lifelong learning flow:**
```
Round 1: Train on seen tasks → Evaluate → Inference (detect unseen samples)
Round 2: Re-train with transferred samples → Evaluate → Final inference
```

The `samples_transfer_ratio` metric measures how many inference samples were identified as coming from an unseen task distribution and transferred for re-training. A non-zero ratio indicates the unseen task detection is working.

---

## System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| OS | Linux (Ubuntu 20.04+) | Ubuntu 22.04 |
| CPUs | 2 | 4+ |
| RAM | 8 GB | 16 GB |
| Disk | 10 GB free | 20 GB free |
| Python | 3.8 | 3.8 |
| GPU | Not required | CUDA-capable GPU |

> **Note:** The benchmark runs on CPU. A GPU will significantly speed up training but is not mandatory.

---

## Step 1 — Ianvs Installation

### 1.1 Clone the repository

```bash
git clone https://github.com/kubeedge/ianvs.git
cd ianvs
```

### 1.2 Create and activate a virtual environment

```bash
python3 -m venv ianvs_env
source ianvs_env/bin/activate
```

### 1.3 Install system dependencies

```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
```

### 1.4 Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 1.5 Install Ianvs

```bash
pip install -e .
```

Verify the installation:

```bash
ianvs --help
```

---

## Step 2 — Dataset Preparation

### 2.1 Download the dataset

```bash
mkdir -p dataset
cd dataset
wget https://kubeedge.obs.cn-north-1.myhuaweicloud.com/ianvs/curb-detection/curb-detection.zip
unzip curb-detection.zip
cd ..
```

After extraction, the dataset directory should look like:

```
dataset/
└── curb-detection/
    ├── train_data/
    │   ├── images/
    │   │   ├── real_aachen_000000_000019_leftImg8bit.png
    │   │   ├── real_aachen_000000_000019_gtFine_labelTrainIds.png
    │   │   └── ...
    │   └── index.txt          # 600 training samples
    └── test_data/
        ├── images/
        │   └── ...
        └── index.txt          # 201 test samples
```

### 2.2 Index file format

Each line in an index file contains two space-separated paths — the RGB image and its corresponding label mask:

```
./images/real_aachen_000000_000019_leftImg8bit.png ./images/real_aachen_000000_000019_gtFine_labelTrainIds.png
./images/sim_00001_leftImg8bit.png ./images/sim_00001_gtFine_labelTrainIds.png
```

- Files prefixed with `real_` are from **Cityscapes** (real camera footage).
- Files prefixed with `sim_` are from **SYNTHIA** (synthetic/simulated).

The `task_definition_by_origin.py` module uses city names in the path (e.g. `aachen`, `berlin`) to automatically classify samples as real or simulated. You do not need to label them separately.

### 2.3 Symlink an existing dataset (alternative)

If the dataset already exists elsewhere on disk:

```bash
ln -sfn /path/to/existing/curb-detection dataset/curb-detection
```

---

## Step 3 — Configuration

All configuration lives in three YAML files.

### 3.1 `benchmarkingjob.yaml`

Located at: `examples/cityscapes-synthia/lifelong_learning_bench/curb-detection/benchmarkingjob.yaml`

Key settings:

| Field | Value | Description |
|---|---|---|
| `workspace` | `./workspace/curb-detection` | Where outputs are written |
| `testenv` | path to `testenv.yaml` | Test environment config |
| `algorithms[0].url` | path to `rfnet_algorithm.yaml` | Algorithm config |
| `sort_by` | `accuracy descend` | Leaderboard sort order |
| `metrics` | `accuracy`, `samples_transfer_ratio` | Metrics shown in results |

### 3.2 `testenv/testenv.yaml`

Located at: `examples/cityscapes-synthia/lifelong_learning_bench/curb-detection/testenv/testenv.yaml`

```yaml
testenv:
  dataset:
    train_index: "./dataset/curb-detection/train_data/index.txt"
    test_index:  "./dataset/curb-detection/test_data/index.txt"

  model_eval:
    model_metric:
      name: "accuracy"
      url: "./examples/cityscapes-synthia/lifelong_learning_bench/curb-detection/testenv/accuracy.py"
    threshold: 0
    operator: "<"

  metrics:
    - name: "accuracy"
      url: "./examples/cityscapes-synthia/lifelong_learning_bench/curb-detection/testenv/accuracy.py"
    - name: "samples_transfer_ratio"

  incremental_rounds: 2
```

### 3.3 `testalgorithms/rfnet/rfnet_algorithm.yaml`

Key hyperparameters under `modules[basemodel].hyperparameters`:

| Parameter | Default | Description |
|---|---|---|
| `learning_rate` | `0.0001` | Adam optimizer learning rate |
| `epochs` | `1` | Training epochs per round |
| `base_size` | `1024` | Image resize base dimension |
| `crop_size` | `768` | Training crop dimension |
| `batch_size` | `4` | Samples per training batch |
| `workers` | `4` | DataLoader worker threads |

---

## Step 4 — Run the Benchmark

Run from the **root of the ianvs repository**. The `PYTHONPATH` must include `sedna_src` and the `RFNet` directory.

### Run command (copy as one line)

```bash
cd /path/to/ianvs && PYTHONPATH=/path/to/ianvs/sedna_src:/path/to/ianvs/examples/cityscapes-synthia/lifelong_learning_bench/curb-detection/testalgorithms/rfnet/RFNet ianvs -f examples/cityscapes-synthia/lifelong_learning_bench/curb-detection/benchmarkingjob.yaml
```

Replace `/path/to/ianvs` with your actual repository path.

> **Important:** Run the command as a single line. Do not press Enter in the middle of the `PYTHONPATH=...` value — splitting it will cause a "No such file or directory" error.

> **Important:** Always run from the repository root, not from inside the example directory. All paths in YAML files are relative to the repo root.

### What happens during a run

```
[Round 1]
  Train  → RFNet trains on 80% of train_index samples
  Eval   → mIoU accuracy computed on test_index
  Infer  → Model predicts on 20% held-out samples; ~50% flagged as unseen

[Round 2]
  Train  → Re-trains with unseen samples transferred
  Eval   → Final mIoU computed
  Infer  → Final inference; samples_transfer_ratio calculated

[Rank]   → Results printed to console and saved to workspace/
```

---

## Step 5 — Reading Results

Results are printed to the console and saved in:

```
workspace/curb-detection/benchmarkingjob/rank/
├── selected_rank.csv    # columns: rank, algorithm, accuracy, samples_transfer_ratio, ...
└── all_rank.csv         # all hyperparameter combinations
```

### Example output

```
| rank | algorithm               | accuracy | samples_transfer_ratio |
|  1   | rfnet_lifelong_learning | 0.2123   | 0.4649                 |
```

### Metric definitions

| Metric | Formula | Meaning |
|---|---|---|
| `accuracy` | mean of (CPA + mIoU + FWIoU) / 3 | Segmentation quality across all classes |
| `samples_transfer_ratio` | `unseen_count / (total_infer + 1)` | Fraction of samples identified as unseen tasks; measures knowledge transfer activity |

- Higher `accuracy` is better.
- A non-zero `samples_transfer_ratio` confirms the unseen task detection pipeline is active.

---

## File Structure

```
examples/cityscapes-synthia/lifelong_learning_bench/curb-detection/
├── README.md                          ← this file
├── benchmarkingjob.yaml               ← top-level benchmark config
├── testenv/
│   ├── testenv.yaml                   ← dataset paths, metrics, eval thresholds
│   └── accuracy.py                    ← mIoU/CPA/FWIoU metric implementation
└── testalgorithms/
    └── rfnet/
        ├── rfnet_algorithm.yaml       ← algorithm + hyperparameter config
        ├── basemodel.py               ← BaseModel: train/predict/evaluate/load/save
        ├── task_definition_by_origin.py  ← splits data into real/sim tasks
        ├── task_allocation_by_origin.py  ← routes inference samples to correct task model
        └── RFNet/                     ← RFNet model implementation
            ├── train.py
            ├── eval.py
            ├── dataloaders/
            │   └── datasets/
            │       └── cityscapes.py  ← dataset loader (handles real + sim)
            └── utils/
                ├── args.py            ← TrainArgs / ValArgs with defaults
                ├── metrics.py         ← confusion matrix + mIoU calculations
                └── summaries.py       ← TensorBoard writer helpers
```

---

## Troubleshooting

### `bash: net/RFNet: No such file or directory`

The command was copy-pasted across multiple lines and the path broke. Run the entire command on **one line**.

---

### `ModuleNotFoundError: No module named 'sedna'`

Either sedna is not installed or `sedna_src` is missing from `PYTHONPATH`.

**Fix A — install the wheel:**
```bash
pip install examples/resources/third_party/sedna-0.6.0.1-py3-none-any.whl
```

**Fix B — use sedna_src (local dev):**
```bash
export PYTHONPATH=/path/to/ianvs/sedna_src:$PYTHONPATH
```

---

### `ModuleNotFoundError: No module named 'RFNet'` or `No module named 'mypath'`

The `RFNet` directory is not in `PYTHONPATH`.

**Fix:**
```bash
export PYTHONPATH=/path/to/ianvs/examples/cityscapes-synthia/lifelong_learning_bench/curb-detection/testalgorithms/rfnet/RFNet:$PYTHONPATH
```

---

### `FileNotFoundError: dataset/curb-detection/...`

The dataset directory does not exist at `./dataset/curb-detection/` relative to the repo root.

**Fix A — download the dataset** (see Step 2).

**Fix B — symlink an existing copy:**
```bash
ln -sfn /path/to/existing/curb-detection dataset/curb-detection
```

---

### `PermissionError: /var/lib/sedna/kb`

Sedna is trying to write its knowledge base to a system path that requires root.

**Fix:** Ensure `sedna_src/sedna/common/constant.py` uses a user-writable path:
```python
import os
EDGE_KB_DIR = os.path.expanduser("~/.sedna/kb")
```

---

### `Connection refused: http://127.0.0.1:9100/sedna/workers/...`

This warning appears because ianvs expects a KubeEdge edge node to be running locally. In a standalone local setup, no edge node is present.

**This is expected and does not block execution.** The benchmark will complete normally despite these warnings.

---

### `ValueError: can't find class type knowledge_management class name UpdateStrategyDefault`

The `UpdateStrategyDefault` class is not registered in sedna 0.6.0.1.

**Fix:** In `sedna_src/sedna/algorithms/seen_task_learning/__init__.py`, add:
```python
from . import task_update_decision
```

And ensure `task_update_decision_finetune.py` contains the `UpdateStrategyDefault` class implementation. See the sedna patches applied in this branch for the full implementation.

---

### `AttributeError: 'TaskDefinitionByOrigin' object has no attribute 'get'`

Sedna's `SeenTaskLearning._task_definition()` expects a dict but ianvs passes a class instance.

**Fix:** In `sedna_src/sedna/algorithms/seen_task_learning/seen_task_learning.py`, add an instance check at the top of `_task_definition()`:
```python
if not isinstance(self.task_definition, dict):
    return self.task_definition(samples, **kwargs)
```

---

### `TypeError: float() argument must be a string or number, not 'dict'`

The scores returned from evaluation are nested dicts; sedna tries to call `float()` on a dict.

**Fix:** In `sedna_src/sedna/core/lifelong_learning/knowledge_management/cloud_knowledge_management.py`, flatten nested scores before the threshold comparison:
```python
flat_scores = []
for v in scores.values():
    if isinstance(v, dict):
        flat_scores.extend(v.values())
    else:
        flat_scores.append(v)
if any(map(lambda x: operator_func(float(x), self.model_threshold), flat_scores)):
```

---

### `samples_transfer_ratio` is always `0.0`

The unseen task detection is not running or is returning `is_unseen_task = False` for all samples.

**Check 1:** Confirm `inference_2()` in `sedna_src/sedna/core/lifelong_learning/lifelong_learning.py` calls `SampleRegonitionDefault` and sets `is_unseen_task = len(unseen_samples.x) > 0`.

**Check 2:** Confirm `SampleRegonitionDefault` is registered via `@ClassFactory.register(ClassType.UTD)` and imported at startup.

---

### Out of memory (OOM) during training

**Fix:** Reduce `batch_size`, `base_size`, and `crop_size` in `rfnet_algorithm.yaml`:
```yaml
- base_size:
    values: [512]
- crop_size:
    values: [384]
- batch_size:
    values: [2]
```

---

### `RuntimeError: value_range` / `range` argument error in TensorBoard

PyTorch >= 2.x renamed the `range` parameter to `value_range` in `make_grid`.

**Fix:** In `RFNet/utils/summaries.py`, replace:
```python
# old
grid_image = make_grid(..., range=(0, 255), ...)
# new
grid_image = make_grid(..., value_range=(0, 255), ...)
```

---

## What is Next

- Raise issues or questions on the [Ianvs GitHub issue tracker](https://github.com/kubeedge/ianvs/issues).
- Explore other lifelong learning examples under `examples/` for cross-benchmark comparison.
- To add a new algorithm, implement the `BaseModel` interface (`train`, `predict`, `evaluate`, `load`, `save`) and register it with `@ClassFactory.register(ClassType.GENERAL, alias="YourModel")`.
