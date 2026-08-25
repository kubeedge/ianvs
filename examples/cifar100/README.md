# CIFAR-100 Federated Learning Benchmarking

This example benchmarks federated learning algorithms on the CIFAR-100 dataset
using the Ianvs benchmarking framework, covering both standard Federated Learning
(FL) and Federated Class-Incremental Learning (FCIL).

## Overview

In real-world edge scenarios, data arrives incrementally across distributed
clients while new classes are introduced over time. This example evaluates
how well federated learning algorithms handle class-incremental learning
without forgetting previously learned classes (catastrophic forgetting).

Two paradigms are covered:

- **`federatedlearning`** — standard federated averaging across distributed
  clients. Used as a baseline before evaluating FCIL algorithms.
- **`federatedclassincrementallearning`** — combines federated learning with
  class-incremental learning. New classes are introduced over time and the
  model must learn them without forgetting previously learned ones.

## Dataset

**CIFAR-100**: 60,000 images across 100 classes (600 images per class).
- 50,000 training images
- 10,000 test images
- Backend: TensorFlow

Prepare the dataset by running the following from the root of the ianvs repository:

```bash
mkdir -p data/cifar100
python examples/cifar100/utils.py
```

This downloads CIFAR-100 via TensorFlow and generates per-class index files
under `data/cifar100/`.

## Algorithms

### Federated Class-Incremental Learning (`fci_ssl/`)

| Algorithm | Directory | Description |
|---|---|---|
| **FedAvg** | `fci_ssl/fedavg/` | Baseline federated averaging adapted for FCIL |
| **GLFC** | `fci_ssl/glfc/` | Global-Local Forgetting Compensation |
| **Fed-CI-Match** | `fci_ssl/fed_ci_match/` | Federated class-incremental matching with pseudo-label semi-supervised learning |
| **Fed-CI-Match-v2** | `fci_ssl/fed_ci_match_v2/` | Improved Fed-CI-Match with learnable local/global convolution blending |

### Standard Federated Learning (baseline)

| Algorithm | Directory | Description |
|---|---|---|
| **FedAvg (FL)** | `federated_learning/fedavg/` | Standard federated averaging without incremental learning |
| **FedAvg (FCIL)** | `federated_class_incremental_learning/fedavg/` | FedAvg adapted for class-incremental setting |

### Sedna Native FL (distributed deployment)

| Directory | Description |
|---|---|
| `sedna_federated_learning/` | Uses KubeEdge Sedna built-in worker architecture with separate train workers and aggregation workers — closer to a real distributed deployment than the ianvs simulation above |

## Key Differences Between Paradigms

| Feature | Standard FL (`federated_learning/`) | FCIL (`fci_ssl/`) |
|---|---|---|
| Classes | Fixed across all rounds | New classes added incrementally |
| Forgetting | Not evaluated | Measured via `forget_rate` |
| Metrics | `accuracy` only | `accuracy` + `task_avg_acc` + `forget_rate` |
| Algorithms | FedAvg only | FedAvg, GLFC, Fed-CI-Match, Fed-CI-Match-v2 |

## Metrics

- **accuracy**: Overall classification accuracy
- **task_avg_acc**: Average accuracy across all incremental tasks
- **forget_rate**: Rate of forgetting previously learned classes

## Prerequisites

- Python >= 3.8
- Ianvs installed (`python setup.py install`)
- KubeEdge Sedna (`pip install resources/third_party/sedna-0.6.0.1-py3-none-any.whl`)
- Dependencies: `pip install -r examples/cifar100/requirements.txt`

## Quick Start

**Step 1 — Prepare dataset:**

```bash
mkdir -p data/cifar100
python examples/cifar100/utils.py
```

**Step 2 — Run a benchmark:**

All commands are run from the root of the ianvs repository.

FCIL algorithms:
```bash
# FedAvg baseline (fastest, recommended starting point)
ianvs -f examples/cifar100/fci_ssl/fedavg/benchmarkingjob.yaml

# GLFC
ianvs -f examples/cifar100/fci_ssl/glfc/benchmarkingjob.yaml

# Fed-CI-Match
ianvs -f examples/cifar100/fci_ssl/fed_ci_match/benchmarkingjob.yaml

# Fed-CI-Match-v2
ianvs -f examples/cifar100/fci_ssl/fed_ci_match_v2/benchmarkingjob.yaml
```

Standard FL baseline:
```bash
ianvs -f examples/cifar100/federated_learning/fedavg/benchmarkingjob.yaml
```

FCIL with federated_class_incremental_learning paradigm:
```bash
ianvs -f examples/cifar100/federated_class_incremental_learning/fedavg/benchmarkingjob.yaml
```

**Step 3 — Check results:**

Results are saved to the `workspace/` directory. A leaderboard is printed
to console showing `task_avg_acc` and `forget_rate` for FCIL algorithms,
and `accuracy` for the standard FL baseline.

## Configuration

Key hyperparameters in each variant's `algorithm/algorithm.yaml`:

| Parameter | Default | Description |
|---|---|---|
| `batch_size` | 64 | Training batch size per client |
| `learning_rate` | 0.001 | Learning rate |
| `epochs` | 1 | Local training epochs per round |
| `train_ratio` | 1.0 | Ratio of data used for training |
| `data_partition` | iid | Data distribution across clients (iid/non-iid) |
| `incremental_rounds` | 2 | Number of incremental learning rounds |
| `client_number` | 1 | Number of federated clients |

## Directory Structure

```
cifar100/
├── fci_ssl/                          # Federated Class-Incremental SSL
│   ├── fedavg/                       # FedAvg baseline for FCIL
│   ├── glfc/                         # GLFC algorithm
│   ├── fed_ci_match/                 # Fed-CI-Match algorithm
│   └── fed_ci_match_v2/              # Fed-CI-Match-v2 algorithm
├── federated_class_incremental_learning/
│   └── fedavg/                       # FedAvg for FCIL paradigm
├── federated_learning/
│   └── fedavg/                       # Standard FedAvg baseline
├── sedna_federated_learning/         # Sedna-native distributed FL
│   ├── train_worker/                 # Client-side training
│   └── aggregation_worker/           # Server-side aggregation
├── utils.py                          # Dataset preparation
└── requirements.txt                  # Python dependencies

```

## Related Issues

- For questions or issues, see the
  [Ianvs issue tracker](https://github.com/kubeedge/ianvs/issues)
