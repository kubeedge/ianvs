# Federated LLM Parameter-Efficient Fine-Tuning (FedLLM-PEFT)

## Overview

This directory contains the implementation of federated fine-tuning for Large Language Models using Parameter-Efficient Fine-Tuning (PEFT) methods within the Ianvs framework. The implementation supports federated learning with LoRA (Low-Rank Adaptation) and P-Tuning techniques, enabling privacy-preserving collaborative training of LLMs across distributed clients.

## Architecture

The FedLLM-PEFT extends Ianvs's federated learning paradigm to support LLM fine-tuning with the following key features:

- **GPU-aware task scheduler**: Efficiently manages multiple clients across available GPUs
- **Parameter-efficient methods**: Supports LoRA and P-Tuning for reduced memory footprint
- **Adaptive aggregation**: FedAvg and FedAvgM algorithms for PEFT parameter aggregation
- **Comprehensive metrics**: ROUGE-1, ROUGE-2, ROUGE-L, and BLEU-4 evaluation metrics

## Directory Structure

```
fedllm-peft/
├── algorithm/
│   ├── algorithm.yaml          # Algorithm configuration
│   ├── model.py               # Base model implementation
│   ├── FedAvg-PEFT.py        # FedAvg aggregation algorithm
│   └── FedAvgM-PEFT.py       # FedAvgM aggregation algorithm
├── testenv/
│   ├── testenv.yaml          # Test environment configuration
│   ├── rouge1_metric.py      # ROUGE-1 evaluation metric
│   ├── rouge2_metric.py      # ROUGE-2 evaluation metric
│   ├── rougel_metric.py      # ROUGE-L evaluation metric
│   └── bleu4_metric.py       # BLEU-4 evaluation metric
├── benchmarkingjob.yaml      # Benchmarking job configuration
└── README.md                 # This file
```

## Supported Models

### Base Models
- **ChatGLM-6B**: `THUDM/chatglm-6b`
- **Other HuggingFace models**: Any transformer-based LLM with adapter support

### PEFT Methods
- **LoRA (Low-Rank Adaptation)**: Efficient fine-tuning through low-rank matrix decomposition
- **P-Tuning**: Continuous prompt learning in embedding space

## Configuration Parameters

### Algorithm Configuration (`algorithm.yaml`)

#### Paradigm Settings
- `paradigm_type`: Set to `"federatedlearning"` for federated learning paradigm
- `fl_data_setting`:
  - `train_ratio`: Proportion of data used for training (default: 1.0)
  - `splitting_method`: Data splitting strategy (default: "default")
  - `label_data_ratio`: Ratio of labeled data (default: 1.0)

#### Base Model Module
- `batch_size`: Training batch size per client (default: 1)
- `learning_rate`: Local learning rate (default: 1e-4)
- `local_epochs`: Number of local training epochs (default: 2)
- `model_name`: HuggingFace model identifier (e.g., "THUDM/chatglm-6b")
- `save_dir`: Directory to save trained models
- `initial_model_url`: Path to initial model weights
- `peft_method`: PEFT technique - "lora" or "ptuning"
- `aggregation`: Aggregation algorithm - "FedAvg-PEFT" or "FedAvgM-PEFT"

Please change the path to `modules_url`, `initial_model_url`, `save_dir` and `aggregation` as per your setup.

#### Aggregation Module (FedAvgM-PEFT)
- `beta`: Momentum factor for server-side momentum (default: 0.7)
- `server_lr`: Server learning rate for aggregation (default: 1.0)

### Test Environment Configuration (`testenv.yaml`)

#### General Settings
- `backend`: Deep learning framework (set to "TORCH")
- `round`: Number of federated learning rounds
- `gpu_num`: Number of available GPUs
- `client_number`: Number of federated clients
- `if_mode_llm`: Enable LLM mode (must be true for FedLLM-PEFT)

#### Dataset Configuration
- `train_data`: Path to training dataset (JSONL format)
- `test_data`: Path to test dataset (JSONL format)

Please change the path to datasets as per your setup.

Datasets:
- **MedicalQA**: Medical question-answering dataset
- **Custom datasets**: Any JSONL format with two fields

#### Evaluation Metrics
- `rouge1_metric`: ROUGE-1 score for text generation quality
- `rouge2_metric`: ROUGE-2 score (bigram overlap)
- `rougel_metric`: ROUGE-L score (longest common subsequence)
- `bleu4_metric`: BLEU-4 score for translation quality

Please change the path to model metrics as per your setup.

### Benchmarking Job Configuration (`benchmarkingjob.yaml`)

- `workspace`: Working directory for experiment outputs
- `testenv`: Path to test environment configuration
- `algorithm`: Path to algorithm configuration
- `test_object`: Specifies algorithms to benchmark
- `rank`: Ranking and visualization settings
  - `sort_by`: Metrics for leaderboard ranking
  - `visualization`: Output format and selected metrics

Please change the path to `testenv.yaml` and `algorithm.yaml` as per your setup.

## Installation and Setup

### Prerequisites
- Python 3.8.18
- PyTorch 2.4.1+cu118
- CUDA-compatible GPU (tested on A100 80GB*4)
- 32GB RAM recommended

### Installation Steps

1. **Clone Ianvs Repository**
   ```bash
   cd ~
   git clone https://github.com/kubeedge/ianvs.git
   ```

2. **Install Dependencies**
   ```bash
   sudo apt update
   sudo apt install libgl1-mesa-glx -y
   cd ~/ianvs
   python -m pip install third_party/*
   python -m pip install -r requirements.txt
   ```

3. **Install Ianvs**
   ```bash
   python setup.py install
   ```

4. **Additional PEFT Dependencies**
   ```bash
   pip install peft transformers datasets rouge_score nltk
   ```

## Usage

### Data Preparation

Prepare your dataset in JSONL format with the following structure:
```json
{"question": "What are the treatments for Abdominal Adhesions ?", "answer": "Abdominal adhesions that do not cause symptoms generally do not require treatment. Surgery is ..."}
```

### Running the Benchmark

Execute the federated LLM fine-tuning benchmark:

```bash
ianvs -f /home/wwh/kubeFed/ianvs/examples/federated-llm/fedllm-peft/benchmarkingjob.yaml
```

Please change the path to `benchmarkingjob.yaml` as per your setup.

### Customization

1. **Adding New Models**: Update `model.py` with your model loading logic
2. **Custom PEFT Methods**: Extend the PEFT configuration in the model module
3. **New Aggregation Algorithms**: Implement custom aggregation methods following the existing pattern such as `FedAvg-PEFT.py`
4. **Additional Metrics**: Add evaluation metrics in the `testenv/` directory