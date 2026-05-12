# Government Benchmark

This example runs the Ianvs single-task learning benchmark for the GovAff
government affairs dataset. It contains two benchmark jobs:

- `objective`: multiple-choice government affairs questions, scored with `acc`
- `subjective`: open-ended government affairs questions, scored by an LLM judge

The commands below assume you run them from the repository root.

## Requirements

- Python 3.9 or 3.10
- Enough disk space for the GovAff dataset and the selected Hugging Face model
- A Kaggle account/API token, or a browser download of the public dataset
- A DeepSeek-compatible API key only if you run the subjective benchmark

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r examples/government/singletask_learning_bench/requirements.txt
```

On Linux or macOS, activate the environment with:

```bash
source .venv/bin/activate
```

## Prepare Dataset

Download the GovAff dataset from Kaggle:

```powershell
python -m pip install kaggle
kaggle datasets download -d kubeedgeianvs/the-government-affairs-dataset-govaff -p dataset
python -m zipfile -e dataset/the-government-affairs-dataset-govaff.zip dataset/govaff_raw
python examples/government/singletask_learning_bench/prepare_dataset.py
```

If you download the zip in a browser, place it under `dataset/`, unzip it to
`dataset/govaff_raw`, then run the same `prepare_dataset.py` command.

After preparation, Ianvs should see this layout:

```text
dataset/government
|-- objective
|   |-- test_data
|   |   |-- data.jsonl
|   |   `-- metadata.json
|   `-- train_data
|       `-- data.jsonl
`-- subjective
    |-- test_data
    |   |-- data.jsonl
    |   `-- metadata.json
    `-- train_data
        `-- data.jsonl
```

The preparation step converts the current Kaggle folder names
`government/multi-choice questions` and `government/subjective questions` into
the paths expected by the Ianvs YAML files.

## Model Configuration

By default, both jobs load `Qwen/Qwen2-0.5B-Instruct` from Hugging Face. To use a
local checkpoint or another compatible causal language model, set:

```powershell
$env:GOVERNMENT_BENCH_MODEL = "C:\path\to\model-or-huggingface-id"
```

On Linux or macOS:

```bash
export GOVERNMENT_BENCH_MODEL=/path/to/model-or-huggingface-id
```

The model uses CUDA when available and falls back to CPU.

## Run

Objective benchmark:

```powershell
ianvs -f examples/government/singletask_learning_bench/objective/benchmarkingjob.yaml
```

Subjective benchmark:

```powershell
$env:DEEPSEEK_API_KEY = "your_api_key"
ianvs -f examples/government/singletask_learning_bench/subjective/benchmarkingjob.yaml
```

Optional subjective judge settings:

```powershell
$env:DEEPSEEK_BASE_URL = "https://api.deepseek.com"
$env:DEEPSEEK_MODEL = "deepseek-chat"
```

Outputs are written under `workspace/government/objective` and
`workspace/government/subjective`.

## Data Format

Metadata files use:

```json
{
    "dataset": "A Objective BenchMark Template",
    "description": "A government benchmark for llm testing",
    "level_1_dim": "single-modal",
    "level_2_dim": "text",
    "level_3_dim": "Q&A",
    "level_4_dim": "government"
}
```

Subjective test rows use the Sedna LLM metadata parser format:

```json
{
    "prompt": "System or task background.",
    "query": "Question text.",
    "response": "Reference answer.",
    "judge_prompt": "Judge prompt ending before the candidate answer.",
    "level_1_dim": "single-modal",
    "level_2_dim": "text",
    "level_3_dim": "Q&A",
    "level_4_dim": "government"
}
```
