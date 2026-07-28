# LLM Edge Benchmark Suite — `single_task_bench`

End-to-end guide for running the Large Language Model (LLM) edge benchmark on the
[KubeEdge-Ianvs](https://github.com/kubeedge/ianvs) framework.

This example runs the **Single Task Learning** paradigm and measures LLM inference
performance — latency, throughput, time-to-first-token (`prefill_latency`), and
memory usage — using [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python)
with a quantized GGUF model (Qwen 1.5 0.5B by default).

> A parallel example with compression metrics lives under
> [`single_task_bench_with_compression/`](../single_task_bench_with_compression/).
> The setup is identical; only the benchmark job and metrics differ.

## 1. Prerequisites

| Requirement | Minimum version | Notes |
|---|---|---|
| OS | Linux x86_64 (tested on Ubuntu 22.04) | macOS works; Windows requires WSL2 |
| Python | 3.8 – 3.10 | Match the version Ianvs supports |
| C/C++ toolchain | gcc/g++ 9+ or clang 12+ | Required to build `llama-cpp-python` |
| RAM | 4 GB free | Larger for bigger models |
| Disk | ~1 GB | Model weights + virtualenv |

Install build tools on Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git wget python3-venv
```

## 2. Clone the repository

```bash
git clone https://github.com/kubeedge/ianvs.git
cd ianvs
export REPO_ROOT="$(pwd)"
```

`$REPO_ROOT` is used throughout this guide. Every other absolute path is derived
from it, so you only need to set it once per shell.

## 3. Create a virtual environment and install Ianvs

```bash
python3 -m venv ianvs_env
source ianvs_env/bin/activate

# Install Ianvs core
python setup.py install

# Install the Sedna client that Ianvs depends on
pip install ./examples/resources/third_party/sedna-0.4.1-py3-none-any.whl
```

Verify the install:

```bash
ianvs --help
```

## 4. Install example-specific dependencies

```bash
pip install -r examples/llm-edge-benchmark-suite/single_task_bench/requirements.txt
```

This pulls in `llama-cpp-python`, `torch`, `transformers`, `pyyaml`, `pandas`,
and `requests`. The `llama-cpp-python` wheel compiles native code on install —
if it fails, double-check the C/C++ toolchain from step 1.

## 5. Prepare the dataset

The benchmark consumes a JSONL file where each line is a `{"question": ..., "answer": ...}`
record. A small sample dataset is committed under `dataset/data.jsonl` at the
repository root.

```bash
mkdir -p "$REPO_ROOT/dataset"
ls "$REPO_ROOT/dataset/data.jsonl"   # should already exist
```

Each line should look like:

```json
{"question": "Which of the following numbers is the smallest prime number?\nA. 0\nB. 1\nC. 2\nD. 4", "answer": "C"}
```

To use your own dataset, drop a JSONL file at the same location or update the
paths in `testenv/testenv.yaml` (see step 7).


## 6. Download the model weights

The default model is **Qwen 1.5 0.5B Chat (Q4_K_M GGUF)**, ~398 MB.

```bash
mkdir -p "$REPO_ROOT/models/qwen"
wget -c \
  -O "$REPO_ROOT/models/qwen/qwen_1_5_0_5b.gguf" \
  "https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat-GGUF/resolve/main/qwen1_5-0_5b-chat-q4_k_m.gguf"
```

Verify the file size — corrupted downloads silently break inference:

```bash
ls -lh "$REPO_ROOT/models/qwen/qwen_1_5_0_5b.gguf"
# expect roughly 398M
```

The `*.gguf` extension is gitignored, so the file stays local and never gets
committed.

## 7. Point the configs at your paths

Two YAML files contain absolute paths that you **must** update for your machine.
Relative paths are not parsed correctly by Ianvs in these fields.

### 7a. `testenv/testenv.yaml`

```yaml
testenv:
  dataset:
    train_data: "<REPO_ROOT>/dataset/data.jsonl"
    test_data:  "<REPO_ROOT>/dataset/data.jsonl"
```

Replace `<REPO_ROOT>` with the value you exported in step 2 (e.g.
`/home/alice/ianvs`).

### 7b. `testalgorithms/algorithm.yaml`

```yaml
algorithm:
  paradigm_type: "singletasklearning"
  initial_model_url: "<REPO_ROOT>/models/qwen/qwen_1_5_0_5b.gguf"
  modules:
    - type: "basemodel"
      name: "LlamaCppModel"
      url: "./examples/llm-edge-benchmark-suite/single_task_bench/testalgorithms/basemodel.py"
      hyperparameters:
        - model_path:
            values:
              - "<REPO_ROOT>/models/qwen/qwen_1_5_0_5b.gguf"
        - n_ctx:
            values:
              - 2048
```

## 8. Algorithm contract (FYI)

`testalgorithms/basemodel.py` implements `LlamaCppModel`. If you fork or extend
it, the Ianvs `SingleTaskLearning` paradigm requires these methods:

| Method | Required by framework? | Purpose |
|---|---|---|
| `preprocess(data=None, **kwargs)` | Yes — called with zero args | No-op pass-through; signature must tolerate the zero-arg call |
| `predict(data, **kwargs)` | Yes | Run inference; uses `stream=True` to capture `prefill_latency` |
| `train(train_data, valid_data=None, **kwargs)` | Yes — called unconditionally | No-op for pre-trained inference; returns the model path |
| `save(model_path)` / `load(model_url)` | Yes | No-ops for GGUF (weights live on disk already) |
| `evaluate(data, **kwargs)` | Yes | Runs `predict` then applies the metric callable from `kwargs["metric"]` |

`postprocess` is **not** called by the SingleTaskLearning paradigm and has been
removed. If you port this model to another paradigm (joint-inference,
incremental-learning) that wraps it in Sedna, add it back as a no-op
pass-through.

Missing optional args on `preprocess` is the most common cause of `TypeError`
during pipeline execution.

## 9. Run the benchmark

From `$REPO_ROOT`:

```bash
ianvs -f examples/llm-edge-benchmark-suite/single_task_bench/benchmarkingjob.yaml
```

Ianvs loads the configs, instantiates `LlamaCppModel`, streams inference over
the test set, and writes results to `./workspace/`.

### Expected output

```text
+------+-----------+---------+------------+-----------------+--------------------+---------------+
| rank | algorithm | latency | throughput | prefill_latency |      paradigm      |   basemodel   |
+------+-----------+---------+------------+-----------------+--------------------+---------------+
|  1   | llama-cpp | 171.29  |   0.0058   |     171.27      | singletasklearning | LlamaCppModel |
+------+-----------+---------+------------+-----------------+--------------------+---------------+
```

Numbers will vary by hardware. The full leaderboard CSV is saved at
`./workspace/benchmarkingjob/rank/all_rank.csv`.

## 10. Metrics

| Metric | Unit | Definition |
|---|---|---|
| `latency` | ms | Mean end-to-end time per prompt |
| `throughput` | tokens/sec | Generated tokens divided by wall time |
| `prefill_latency` | ms | Time-to-first-token (TTFT) — measured via the first streamed chunk |
| `mem_usage` | bytes | Resident set size of the inference process |

Metric implementations live in `testenv/*.py`.


## 11. Next steps

- Swap in another GGUF model by changing `initial_model_url` and the
  `model_path` hyperparameter in `algorithm.yaml`.
- Enable GPU offload by setting `n_gpu_layers` in `algorithm.yaml` (requires a
  CUDA-enabled `llama-cpp-python` build).
- Try the compression variant under
  [`single_task_bench_with_compression/`](../single_task_bench_with_compression/)
  to compare quantization strategies.
