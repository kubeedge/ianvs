# llm-edge-benchmark-suite single_task_bench_with_compression

This guide outlines the complete setup, configuration, and execution process for running the **Compression** Large Language Model (LLM) benchmarking suite using the [Ianvs](https://github.com/kubeedge/ianvs) edge computing framework.

>  **CRITICAL FIRST STEP: DEPENDENCIES**
> You must install the necessary packages via `requirements.txt` before executing any runs.

---

## Step 1: Environment Setup

First, ensure your Ianvs virtual environment is active:
```bash
source /path/to/your/ianvs_env/bin/activate
```

Install the requirements.txt
```bash
pip install -r examples/llm-edge-benchmark-suite/single_task_bench_with_compression/requirements.txt
```

---

##  Step 2: Shared Model Acquisition

This benchmark shares the same `.gguf` model file as the standard suite. Ensure the `Qwen1.5-0.5B-Chat` model exists in your central models directory.

If it is missing, download it using a resumable command:
```bash
mkdir -p models/qwen
wget -c -O models/qwen/qwen_1_5_0_5b.gguf https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat-GGUF/resolve/main/qwen1_5-0_5b-chat-q4_k_m.gguf
```

---

##  Step 3: Configuration

`testenv/testenv.yaml` already points to the minimal dataset bundled with this example
(`dataset/train_data/data.jsonl` and `dataset/test_data/data.jsonl`), so no path changes
are needed to run the benchmark out of the box.

##  Step 4: Execution

Execute the benchmark:

```bash
ianvs -f examples/llm-edge-benchmark-suite/single_task_bench_with_compression/benchmarkingjob.yaml
```

### Expected Output
Ianvs will execute the benchmark and generate a `workspace` directory. You will see a successful run log and a final table detailing latency, throughput, and prefill latency.