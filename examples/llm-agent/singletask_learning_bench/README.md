# Ianvs LLM-Agent Benchmark — singletask_learning_bench

This example fine-tunes and evaluates a language model on a small activity-classification dataset. Ianvs loads `Langboat/bloom-1b4-zh`, applies LoRA fine-tuning, generates predictions, and reports ROUGE-1, ROUGE-2, and ROUGE-L scores.

To run the example, follow [Quick start](#2-quick-start).

> The default configuration uses the same 10 samples for training and testing. This demonstrates the complete pipeline and may measure memorisation. It does not measure performance on unseen data.

## 1. Prerequisites

- Python 3.10 in an isolated virtual environment.
- Sufficient RAM to hold the model and the additional memory required for training. The CPU workflow below completed on a 16 GB RAM GitHub Codespace.
- Disk space for dependencies, downloaded model weights, checkpoints, and benchmark outputs.
- Network access to install dependencies and download the model on the first run.

Run all commands from the Ianvs repository root unless stated otherwise.

The validated CPU run loaded the model in float32. On an 8 GB Codespace, automatic device placement offloaded some layers to disk and training failed during the backward pass. On the 16 GB machine, all model parameters remained on CPU and the complete benchmark passed. This is an observed configuration, not a measured minimum RAM requirement.

## 2. Quick start

### 2.1 Create a Python environment

If you do not already have a checkout:

```bash
git clone https://github.com/kubeedge/ianvs.git
cd ianvs
```

Create and activate a Python 3.10 virtual environment:

```bash
python3.10 -m venv ianvs_env
source ianvs_env/bin/activate
```
### 2.2 Install PyTorch for CPU or GPU

Choose one of the following PyTorch installation options, depending on your environment. Install PyTorch before installing the repository and example dependencies.

**CPU-only**

For CPU-only environments, install the CPU build of PyTorch:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**GPU (CUDA 12.6)**

For a machine with a compatible NVIDIA GPU and driver, install the CUDA-enabled PyTorch build:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

For other CUDA versions, consult the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) and select the appropriate installation command for your operating system and hardware.

**Verify the installation**

Run the following command to check whether PyTorch can access a CUDA-enabled GPU:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not available')"
```

If `CUDA available` is `False`, check that the machine has a compatible NVIDIA GPU, that its drivers are installed, and that the selected PyTorch build supports the environment. Installing CUDA-enabled PyTorch alone does not provide GPU hardware.

**Install Ianvs**

After installing the appropriate PyTorch build, install the repository dependencies and Ianvs:

```bash
pip install -r requirements.txt
pip install .
```

The benchmark documented in this README was validated on CPU. GPU execution has not yet been validated for this example.

### 2.3 Install the bundled KubeEdge Sedna wheel

Install the wheel included with Ianvs:

```bash
pip install resources/third_party/sedna-0.6.0.1-py3-none-any.whl
```

Use the bundled KubeEdge Sedna wheel for this example. A separate Sedna clone-and-build step is not required. Its declared dependencies were installed automatically during the validated setup.

### 2.4 Install the example dependencies

```bash
pip install -r examples/llm-agent/singletask_learning_bench/requirements.txt
pip check
```

Resolve any incompatibilities reported by `pip check` before running the benchmark. A successful dependency check confirms consistency of installed package requirements; the benchmark run verifies runtime behaviour.

### 2.5 Run the benchmark

The repository includes the 10-sample dataset at `examples/llm-agent/dataset/activity_classification.jsonl`. No dataset-generation command is required.

```bash
ianvs -f examples/llm-agent/singletask_learning_bench/benchmarkingjob.yaml
```

The first run downloads the configured model. Later runs can reuse the Hugging Face cache, normally under `~/.cache/huggingface` unless configured otherwise.

Ianvs trains the model, generates predictions, computes metrics, and prints a leaderboard. The job configuration writes benchmark outputs under `./workspace`; training checkpoints use `./checkpoint` with the supplied training configuration.

### 2.6 Validated CPU run

The complete workflow passed in a fresh Python 3.10 environment using the bundled Sedna wheel and the existing model and training configurations, including `device: "auto"`.

| Component | Validated version or configuration |
|---|---|
| Hardware | GitHub Codespace with 16 GB RAM; CPU-only execution |
| Python | 3.10.22 |
| Ianvs | 0.3.0 |
| Sedna | 0.6.0.1 |
| PyTorch | 2.14.1+cpu |
| Transformers | 5.20.0 |
| datasets | 2.21.0 |
| huggingface-hub | 1.33.0 |
| PEFT | 0.21.2 |
| Accelerate | 1.15.0 |

`pip check` reported no broken requirements. Model inspection showed all 293 parameter tensors on CPU in float32, with no parameters on the meta device.

Training completed 40 steps over 20 epochs in approximately 9 minutes 25 seconds. Prediction and scoring completed, and the benchmark exited with code `0`.

| Metric | Observed score (0–10 scale) |
|---|---:|
| ROUGE-1 | 10.0 |
| ROUGE-2 | 0.0 |
| ROUGE-L | 10.0 |

These are observed results, not guaranteed outputs or a dependency lockfile. Runtime and scores can vary with hardware, resolved package versions, and training results. See [How the example works](#5-how-the-example-works) for metric interpretation.

## 3. Dataset

### 3.1 Format

The example uses JSON Lines (JSONL): one JSON object per line, with a `question` string and an `answer` string.

```jsonl
{"question": "What activity is the user performing? User is moving rapidly on foot across the track.", "answer": "Running"}
{"question": "What activity is the user performing? User is sitting perfectly still in a chair.", "answer": "Resting"}
{"question": "What activity is the user performing? User is pedaling a two-wheeled vehicle.", "answer": "Cycling"}
```

Use the `.jsonl` extension so Ianvs selects the JSONL loading path. Do not wrap the records in a JSON array.

### 3.2 Bundled sample dataset

The default dataset contains 10 activity-classification samples:

```text
examples/llm-agent/dataset/activity_classification.jsonl
```

The default `testenv.yaml` uses this file for both training and testing. You do not need to create it manually or run a generator.

### 3.3 Using your own dataset

Prepare JSONL records with the same `question` and `answer` fields. To evaluate performance on unseen examples, create separate training and test files and update these fields in `singletask_learning_bench/testenv/testenv.yaml`:

- `dataset.train_data`
- `dataset.test_data`

Ianvs loads the datasets and passes them to the model implementation. The `data_dir` entry in `config/config.json` is not used by the current implementation to load data.

Run the benchmark from the repository root so relative paths resolve correctly.

## 4. Configuration files

### 4.1 `singletask_learning_bench/benchmarkingjob.yaml`

The top-level job configuration identifies the test environment and algorithm, sets the benchmark workspace, and configures the leaderboard. Its selected metrics are `rouge1`, `rouge2`, and `rougeL`.

### 4.2 `singletask_learning_bench/testenv/testenv.yaml`

Defines the dataset paths and metric implementations:

- `dataset.train_data` identifies the training dataset.
- `dataset.test_data` identifies the test dataset.
- `metrics[].url` points to the metric implementation in `rouge.py`.

Both dataset paths point to the bundled sample file by default.

### 4.3 `singletask_learning_bench/testalgorithms/test_algorithm.yaml`

Identifies `basemodel.py` and the model and training configuration files passed to it.

### 4.4 `config/config.json`

The supplied configuration is:

```json
{
    "tokenizer_dir": "Langboat/bloom-1b4-zh",
    "auth_token": null,
    "data_dir": "./examples/llm-agent/dataset/activity_classification.jsonl",
    "token_factor": 32,
    "half_model": true,
    "token_padding": "right",
    "trust_remote": true,
    "device": "auto",
    "output_dir": "./checkpoint"
}
```

The current model implementation uses `tokenizer_dir` and `auth_token` when loading the model and tokenizer. It passes `device` as the model loader's `device_map` argument and `trust_remote` as `trust_remote_code`.

Several fields are present without affecting the current execution path:

- `data_dir` does not load the datasets; Ianvs uses the paths in `testenv.yaml`.
- `half_model` does not set the model's precision. The validated CPU run loaded float32 weights despite this field being `true`.
- `token_padding` is not applied by the current implementation.
- `token_factor` is stored on the model instance but is not used by the shown training or prediction methods.
- `output_dir` in this file does not control training checkpoints. The training configuration supplies that setting.

With `device: "auto"`, placement depends on available memory. Inspect the model's device map and parameter devices if training fails; see [Troubleshooting](#6-troubleshooting).

### 4.5 `config/train_config.json`

The supplied training configuration is:

```json
{
    "per_device_train_batch_size": 5,
    "logging_steps": 50,
    "num_train_epochs": 20,
    "output_dir": "./checkpoint",
    "half_lora": "True",
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "save_strategy": "epoch",
    "save_total_limit": 10
}
```

`half_lora` is an example-specific setting removed before the other fields are passed to `transformers.TrainingArguments`. Its current value is the string `"True"`, not the JSON boolean `true`. Because `basemodel.py` checks equality with the boolean `True` before calling `.half()`, this string does not activate that conversion. The successful run used the configuration as supplied.

Adjust batch size, epochs, learning rate, and checkpoint settings for your dataset and hardware. Changes to precision handling require corresponding implementation validation.

## 5. How the example works

1. **Load configuration and model.** Ianvs reads the job configuration and invokes the example's model implementation, which loads the model and tokenizer.
2. **Load and preprocess data.** Ianvs loads the JSONL records. `basemodel.py` formats the questions and answers for causal language-model training, masks prompt tokens in the labels, and limits training sequences to 128 tokens.
3. **Train.** The example adds LoRA adapters and trains using the configured training settings.
4. **Predict.** For each test question, the model generates up to eight new tokens. The current implementation keeps the first whitespace-separated word of the decoded response.
5. **Score.** `rouge.py` compares the predictions with the reference answers and averages the per-sample ROUGE F1 scores.

The bundled dataset uses single-word activity labels. Review the generation limit and first-word truncation before adapting the example to longer answers.

### Understanding the scores

Each metric function multiplies its average F1 score by `10`, so the leaderboard uses a **0–10 scale**:

| Metric | What it measures |
|---|---|
| ROUGE-1 | Overlap of individual words |
| ROUGE-2 | Overlap of consecutive two-word sequences |
| ROUGE-L | Similarity based on the longest common subsequence of words |

The scorer uses stemming. A score of `10.0` corresponds to an average F1 of `1.0` under that metric, rather than necessarily identical raw strings. These ROUGE scores are not classification accuracy.

ROUGE-2 is `0.0` in the documented run because the single-word reference labels contain no two-word sequences to match. This score alone does not indicate a failed pipeline.

The default training and test sets are identical. Use separate datasets before interpreting the results as evidence of performance on unseen examples.

## 6. Troubleshooting

| Symptom | What to check |
|---|---|
| Test-environment configuration cannot be found | Run from the repository root and check the paths in `benchmarkingjob.yaml`. |
| Dataset file cannot be found | Check `dataset.train_data` and `dataset.test_data` in `testenv.yaml`, and confirm the referenced files exist. |
| JSON parsing fails | Confirm each nonempty line is a valid JSON object with the expected fields. |
| An import from Sedna fails | Confirm that the bundled KubeEdge Sedna wheel is installed in the active environment. |
| `pip check` reports incompatible packages | Inspect the reported requirements and resolve the installation conflict before running the benchmark. |
| Backward pass fails with `expected device meta but got cpu` | Inspect `model.hf_device_map` and parameter devices. The observed failure involved automatic disk offloading and meta parameters on an 8 GB Codespace. A 16 GB run kept all parameters on CPU and completed. Provide enough memory for both weights and training; forcing CPU placement alone does not reduce memory requirements. |
| CPU run warns that `pin_memory` cannot be used without an accelerator | This warning did not prevent the validated CPU run from completing. Check subsequent output for an actual error. |
| Training or prediction runs out of memory | Review model size, batch size, precision handling, and available memory. Changing an unused configuration field will not change model precision. |
| Predictions are empty or unexpected | Inspect generated responses, training results, and prompt formatting. Review the first-word truncation in `predict()`. |
| The model cannot be downloaded | Check network access, model availability, and any required authentication. |

## 7. Adapting the example

- **Separate training and testing.** Use different files to evaluate performance on unseen examples.
- **Use representative data.** Include questions and answers that reflect your intended task.
- **Adjust training settings.** Select batch size, learning rate, and epochs for your data and resources.
- **Handle longer answers.** Review the 128-token input limit, eight-token generation limit, and first-word truncation in `basemodel.py`.
- **Choose suitable metrics.** ROUGE measures text overlap. Consider whether it captures success for your task; exact-match accuracy may be useful for classification labels.
- **Use a compatible model.** Verify tokenizer, prompt formatting, LoRA configuration, and memory requirements when changing the model.

## 8. File layout

| Path under `examples/llm-agent/` | Purpose |
|---|---|
| `config/config.json` | Model and device configuration, including some currently unused fields |
| `config/train_config.json` | Training and checkpoint settings |
| `dataset/activity_classification.jsonl` | Bundled 10-sample dataset |
| `singletask_learning_bench/benchmarkingjob.yaml` | Benchmarking job configuration |
| `singletask_learning_bench/requirements.txt` | Example dependencies |
| `singletask_learning_bench/README.md` | Setup and usage instructions |
| `singletask_learning_bench/testalgorithms/basemodel.py` | Model loading, preprocessing, training, and prediction |
| `singletask_learning_bench/testalgorithms/test_algorithm.yaml` | Algorithm configuration |
| `singletask_learning_bench/testenv/rouge.py` | ROUGE metric implementations |
| `singletask_learning_bench/testenv/testenv.yaml` | Dataset and metric configuration |
