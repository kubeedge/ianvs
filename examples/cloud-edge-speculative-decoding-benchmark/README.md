# Cloud-Edge Speculative Decoding Simulation Benchmark

## Table of Contents

- [Introduction](#introduction)
- [Supported Algorithms](#supported-algorithms)
- [Quick Start](#quick-start)
  - [1. Prepare the Environment](#1-prepare-the-environment)
  - [2. Prepare Datasets](#2-prepare-datasets)
  - [3. Select a Dataset](#3-select-a-dataset)
  - [4. Configure Models and Runtime Options](#4-configure-models-and-runtime-options)
  - [5. Run the Benchmark](#5-run-the-benchmark)
- [Metrics and Rank Output](#metrics-and-rank-output)
- [Extending the Example](#extending-the-example)
- [Demo](#demo)
- [Notes](#notes)

## Introduction

This example implements a cloud-edge speculative decoding simulation benchmark on top of the Ianvs `jointinference` paradigm. It provides complete benchmark entries, dataset preparation scripts, metric implementations, runtime utilities, and algorithm implementations for evaluating speculative decoding under an Ianvs workflow.

## Supported Algorithms

This example contains two speculative decoding algorithm implementations.

| Algorithm | Description | Supported Modes |
| --- | --- | --- |
| `AR` | Token-level autoregressive speculative decoding | `edge-only`, `cloud-only`, `collaboration` |
| `Block` | dFlash-style block speculative decoding | `cloud-only`, `collaboration` |

The `Block` algorithm does not provide an `edge-only` baseline because the block drafter depends on target-side hidden states.

## Quick Start

Run all commands from the Ianvs repository root unless stated otherwise.

### 1. Prepare the Environment

This example requires Python `3.10+` and an NVIDIA GPU for the larger model configurations.

```bash
git clone https://github.com/kubeedge/ianvs.git
cd ianvs

conda create -n ianvs-specdec python=3.10 -y
conda activate ianvs-specdec

pip install resources/third_party/sedna-0.6.0.1-py3-none-any.whl
pip install -e .
pip install -r examples/cloud-edge-speculative-decoding-benchmark/requirements.txt
```

Install a PyTorch build that matches your local CUDA and driver stack. See the PyTorch installation guide if the default wheel is not suitable for your machine.

If Hugging Face access requires a mirror, set the endpoint before preparing datasets or running the benchmark:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HUGGINGFACE_HUB_BASE_URL=https://hf-mirror.com
```

### 2. Prepare Datasets

This example can run any text-generation dataset that follows the Ianvs `test_data` layout. A helper script is provided for preparing `gsm8k`, `humaneval`, and `mt_bench` as ready-to-run examples:

```bash
python examples/cloud-edge-speculative-decoding-benchmark/scripts/prepare_benchmark_datasets.py \
  --dst-root examples/cloud-edge-speculative-decoding-benchmark/dataset
```

To prepare only selected example datasets:

```bash
python examples/cloud-edge-speculative-decoding-benchmark/scripts/prepare_benchmark_datasets.py \
  --dst-root examples/cloud-edge-speculative-decoding-benchmark/dataset \
  --datasets gsm8k humaneval
```

Custom datasets should use the same layout:

```text
examples/cloud-edge-speculative-decoding-benchmark/dataset/<dataset_name>/
├── train_data/
│   └── data.jsonl
└── test_data/
    ├── data.jsonl
    └── metadata.json
```

A single-turn `data.jsonl` row should contain at least `query` and `response`:

```json
{"query": "Solve 3x + 11 = 29.", "response": "x = 6", "level_3_dim": "math", "level_4_dim": "custom_math"}
```

Optional fields such as `request_id`, `task_name`, `question_id`, `category`, `prompt_tokens`, and `completion_tokens` are preserved when present.

A multi-turn dataset should be written as ordered turn-level rows. Each row stores the current user turn in `query` and includes `conversation_id`, `turn_index`, and `turn_count` in the JSON-encoded query payload:

```json
{"query": "{\"request_id\":\"conv-001-turn-0\",\"query\":\"Write a short travel plan for Paris.\",\"task_name\":\"custom_chat\",\"conversation_id\":\"conv-001\",\"turn_index\":0,\"turn_count\":2}", "response": "", "level_3_dim": "chat", "level_4_dim": "custom_chat"}
{"query": "{\"request_id\":\"conv-001-turn-1\",\"query\":\"Rewrite it in a more formal style.\",\"task_name\":\"custom_chat\",\"conversation_id\":\"conv-001\",\"turn_index\":1,\"turn_count\":2}", "response": "", "level_3_dim": "chat", "level_4_dim": "custom_chat"}
```

`metadata.json` can follow the standard Ianvs dimension metadata format used by other examples.

### 3. Select a Dataset

The active dataset is configured in:

- `examples/cloud-edge-speculative-decoding-benchmark/testenv/testenv.yaml`

Example:

```yaml
testenv:
  dataset:
    train_data: "./examples/cloud-edge-speculative-decoding-benchmark/dataset/gsm8k/train_data/data.jsonl"
    test_data_info: "./examples/cloud-edge-speculative-decoding-benchmark/dataset/gsm8k/test_data/metadata.json"
```

To use another dataset, point `train_data` and `test_data_info` to that dataset's local files.

### 4. Configure Models and Runtime Options

The `AR` algorithm configuration is:

- `examples/cloud-edge-speculative-decoding-benchmark/testalgorithms/speculative-decoding/test_speculative_decoding.yaml`

Example `AR` settings:

```yaml
drafter:
  inference_mode: "collaboration"
  draft_tokens_per_step: 8
  model: "Qwen/Qwen2.5-0.5B-Instruct"

verifier:
  model: "Qwen/Qwen2.5-7B-Instruct"
```

The `Block` algorithm configuration is:

- `examples/cloud-edge-speculative-decoding-benchmark/testalgorithms/speculative-decoding/test_speculative_decoding_block.yaml`

Example `Block` settings:

```yaml
drafter:
  inference_mode: "collaboration"
  device: "auto"
  draft_tokens_per_step: 16
  model: "z-lab/Qwen3-8B-DFlash-b16"

verifier:
  device: "auto"
  draft_tokens_per_step: 16
  model: "Qwen/Qwen3-8B"
```

Shared runtime options are configured in:

- `examples/cloud-edge-speculative-decoding-benchmark/testalgorithms/speculative-decoding/profiles/base.yaml`

Common options include:

| Option | Meaning |
| --- | --- |
| `prompt_tokens` | Maximum prompt token length |
| `max_new_tokens` | Maximum generated token count |
| `sample_temperature` | Sampling temperature |
| `draft_tokens_per_step` | Draft length per speculative decoding round |
| `device` | Model device, for example `auto`, `cuda:0`, or `cuda:1` |
| `enable_network_sleep` | Whether to sleep for simulated network delay |
| `network_rtt_ms` | Simulated round-trip time |
| `network_uplink_bandwidth_mbps` | Simulated edge-to-cloud bandwidth |
| `network_downlink_bandwidth_mbps` | Simulated cloud-to-edge bandwidth |
| `attn_implementation` | Transformers attention backend, for example `sdpa` |

### 5. Run the Benchmark

Run the `AR` benchmark:

```bash
ianvs -f examples/cloud-edge-speculative-decoding-benchmark/benchmarkingjob.yaml
```

Run the `Block` benchmark:

```bash
ianvs -f examples/cloud-edge-speculative-decoding-benchmark/benchmarkingjob_block.yaml
```

Benchmark outputs are written under the workspace configured in each benchmarking job file:

- `workspace-cloud-edge-speculative-decoding-benchmark-ar`
- `workspace-cloud-edge-speculative-decoding-benchmark-block`

## Metrics and Rank Output

This example reports:

| Metric | Meaning |
| --- | --- |
| `Time to First Token` | Average time to generate the first output token |
| `Throughput` | Average end-to-end output throughput |
| `Internal Token Latency` | Average decode-stage token latency |
| `End-to-End Latency` | Average request latency |
| `Acceptance Rate` | Accepted draft-token ratio for collaboration mode |

Metric implementations are located in:

- `examples/cloud-edge-speculative-decoding-benchmark/testenv/`

The benchmark writes both full and compact rank files:

- `all_rank.csv` keeps full testcase metadata and all hyperparameters for traceability.
- `selected_rank.csv` keeps the compact columns configured by the benchmark job for display.

The compact rank output keeps the main model, mode, sample size, draft length, and metric fields, while omitting local workspace paths and framework module names.

## Extending the Example

New speculative decoding algorithms should keep the Ianvs module layout:

```text
testalgorithms/speculative-decoding/algorithms/<algorithm_name>/
├── drafter.py
└── verifier.py
```

Algorithm code should inherit from the shared base classes:

- `BaseSpeculativeDrafter`
- `BaseSpeculativeVerifier`

The algorithm-specific drafter implements a decorated `step` method:

```python
from common.decorators import specdec_draft
from common.schema import DraftResult


@specdec_draft
def step(self, session, *, window, feedback=None):
    return DraftResult(
        draft_ids=draft_ids,
        data={"algorithm_private_state": state},
        edge_compute_ms=edge_compute_ms,
    )
```

The algorithm-specific verifier implements a decorated `verify` method:

```python
from common.decorators import specdec_verify
from common.payload import control_payload, token_payload
from common.schema import VerifyResult


@specdec_verify
def verify(self, session, *, draft_output, draft_ids):
    return VerifyResult(
        accepted_ids=accepted_ids,
        corrected_ids=corrected_ids,
        rejected_draft_ids=rejected_draft_ids,
        payloads=[
            token_payload(draft_ids, "edge_to_cloud", "draft_ids"),
            token_payload(accepted_ids, "cloud_to_edge", "accepted_ids"),
            token_payload(corrected_ids, "cloud_to_edge", "corrected_ids"),
            control_payload("cloud_to_edge", "round_control"),
        ],
        data={"algorithm_private_result": result},
        cloud_compute_ms=cloud_compute_ms,
        stop=stop,
        stop_reason=stop_reason,
    )
```

The shared runtime handles:

- request normalization
- session state
- timing aggregation
- network payload byte accounting
- optional network sleep simulation
- feedback routing between drafter and verifier
- Ianvs response formatting
- benchmark metric parsing

Algorithm code only needs to return typed `DraftResult` and `VerifyResult` objects. Custom algorithm data can be placed in the `data` field and will be preserved in the payload passed through the runtime.

A model-free reference template is provided in:

- `examples/cloud-edge-speculative-decoding-benchmark/testalgorithms/speculative-decoding/algorithms/template/`

## Demo

https://github.com/user-attachments/assets/355735fc-716b-4b13-ae9a-dedf9101208f

This demo video is 20x accelerated.

Example console output:

```text
+------+----------------------------+---------------------+------------+------------------------+--------------------+-----------------+-------------------------------+------------------------+---------------------------+----------------+-------------------------------+--------------------------------+
| rank |         algorithm          | Time to First Token | Throughput | Internal Token Latency | End-to-End Latency | Acceptance Rate | dataset_processor-sample_size | drafter-inference_mode |       drafter-model       | verifier-model | drafter-draft_tokens_per_step | verifier-draft_tokens_per_step |
+------+----------------------------+---------------------+------------+------------------------+--------------------+-----------------+-------------------------------+------------------------+---------------------------+----------------+-------------------------------+--------------------------------+
|  1   | speculative-decoding-block |        0.528        |   10.62    |         0.0964         |       25.117       |      0.419      |               5               |     collaboration      | z-lab/Qwen3-8B-DFlash-b16 | Qwen/Qwen3-8B  |               16              |               16               |
|  2   | speculative-decoding-block |        0.627        |    1.92    |         0.5216         |      133.629       |                 |               5               |       cloud-only       | z-lab/Qwen3-8B-DFlash-b16 | Qwen/Qwen3-8B  |               16              |               16               |
+------+----------------------------+---------------------+------------+------------------------+--------------------+-----------------+-------------------------------+------------------------+---------------------------+----------------+-------------------------------+--------------------------------+
```

Actual numbers depend on dataset, model pair, GPU, software stack, and runtime configuration.

## Notes

- Run the benchmark from the Ianvs repository root.
- The first run downloads models from Hugging Face unless they are already cached.
- The `Block` path follows a dFlash-style block speculative decoding workflow.
- Enable `enable_network_sleep` only when simulated network delay should be included in measured latency.
- For quick smoke tests, reduce `sample_size`, `warmup_samples`, and `max_new_tokens` in the YAML configuration files.
