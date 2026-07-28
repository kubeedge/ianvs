# llm-edge-benchmark-suite single_task_bench_with_compression

This guide outlines the complete setup, configuration, and execution process for running the **Compression** Large Language Model (LLM) benchmarking suite using the [Ianvs](https://github.com/kubeedge/ianvs) edge computing framework.

>  **CRITICAL FIRST STEPS: ABSOLUTE PATHS & DEPENDENCIES**
> 1. **Correct all paths:** You **must** change every relative path (e.g., `models/qwen/...` or `dataset/...`) in all your `.yaml` configuration files to **absolute paths** (e.g., `/home/user/ianvs/models/qwen/...`). Ianvs will crash if it encounters relative paths.
> 2. **Dependencies:** You must install the necessary packages via `requirements.txt` before executing any runs.

---

## Step 1: Environment Setup

First, ensure your Ianvs virtual environment is active:
```bash
source /path/to/your/ianvs_env/bin/activate
```

Install the requirements.txt
```bash
pip install -r ianvs/examples/llm-edge-benchmark-suite/single_task_bench_with_compression/requirements.txt
```

---

##  Step 2: Shared Model Acquisition

This benchmark shares the same `.gguf` model file as the standard suite. Ensure the `Qwen1.5-0.5B-Chat` model exists in your central models directory.

If it is missing, download it using a resumable command:
```bash
mkdir -p /ianvs/models/qwen
wget -c -O ianvs/models/qwen/qwen_1_5_0_5b.gguf [https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat-GGUF/resolve/main/qwen1_5-0_5b-chat-q4_k_m.gguf](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat-GGUF/resolve/main/qwen1_5-0_5b-chat-q4_k_m.gguf)
```

---

##  Step 3: Configuration Alignment (Fixing the YAMLs)

You must manually update three different YAML files to remove relative paths and fix framework naming strictness.

### 1. Test Environment (`testenv/testenv.yaml`)
Update the dataset location to an absolute path:
```yaml
dataset:
  train_data: "ianvs/dataset/data.jsonl"
```

##  Step 4: Execution

Once all paths are absolute and the script is updated, execute the benchmark:

```bash
ianvs -f ianvs/examples/llm-edge-benchmark-suite/single_task_bench_with_compression/benchmarkingjob.yaml
```

### Expected Output
Ianvs will execute the benchmark and generate a `workspace` directory. You will see a successful run log and a final table detailing latency, throughput, and prefill latency.