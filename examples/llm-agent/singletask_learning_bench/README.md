# Ianvs LLM-Agent Benchmark — singletask_learning_bench

End-to-end benchmark of an LLM agent on a small activity-classification task. The pipeline loads `Langboat/bloom-1b4-zh`, LoRA-fine-tunes it on a JSONL dataset, generates predictions, and reports ROUGE-1 / ROUGE-2 / ROUGE-L.

If you just want to run it, jump to [Quick start](#quick-start). The rest of the doc explains what each piece is and how to change it.

---

## 1. Prerequisites

- Linux or macOS (tested on Ubuntu 22.04)
- Python 3.10 or 3.12
- ~8 GB RAM free, ~6 GB disk for the model weights (downloaded automatically on first run by HuggingFace)
- Network access to `huggingface.co` for the first run

---

## 2. Quick start

```bash
# 1. Clone and enter the repo
git clone https://github.com/kubeedge/ianvs.git
cd ianvs

# 2. Create a venv and install ianvs itself
python3 -m venv ianvs_env
source ianvs_env/bin/activate
pip install -r requirements.txt
pip install .

# 3. Install the ML deps used by this example
pip install -r examples/llm-agent/singletask_learning_bench/requirements.txt

# 4. Create the dataset file (the dataset folder is gitignored, so you have to make it yourself)
#    See section 3 below for the format. A minimal 10-line file is enough to confirm the
#    pipeline works.

# 5. Run the benchmark
ianvs -f examples/llm-agent/singletask_learning_bench/benchmarkingjob.yaml
```

First run is slow because HuggingFace downloads `Langboat/bloom-1b4-zh` (~3 GB) into `~/.cache/huggingface`. Subsequent runs use the cache and start immediately.

Expected output (with the default config — 20 epochs over a 10-sample file where train == test):

```
+------+-----------+--------+--------+--------+--------------------+
| rank | algorithm | rouge1 | rouge2 | rougeL |      paradigm      |
+------+-----------+--------+--------+--------+--------------------+
|  1   | LLM_agent |  10.0  |  0.0   |  10.0  | singletasklearning |
+------+-----------+--------+--------+--------+--------------------+
```

`rouge2 = 0` is expected — the labels in the sample dataset are single words, so there are no bigrams to score.

> The 10.0 on rouge1/L is essentially memorisation: `train_data` and `test_data` point at the same file. To do a real evaluation, split your data into separate train and test JSONL files and update `testenv.yaml`.

---

## 3. Dataset

### 3.1 Format

The `JsonlDataParse` class in sedna hardcodes the keys `question` and `answer`. Each line of the file must be a valid JSON object with those two keys:

```jsonl
{"question": "What activity is the user performing? User is moving rapidly on foot across the track.", "answer": "Running"}
{"question": "What activity is the user performing? User is sitting perfectly still in a chair.", "answer": "Resting"}
{"question": "What activity is the user performing? User is pedaling a two-wheeled vehicle.", "answer": "Cycling"}
```

Use `.jsonl` extension (not `.json`). Ianvs picks the parser by file extension.

### 3.2 Where to put it

```
examples/llm-agent/dataset/activity_classification.jsonl
```

The `dataset/` folder is in `.gitignore` so you won't see it in `git status` — create it locally.

### 3.3 Quick generator

If you want to bootstrap from a flat list:

```bash
python3 - <<'EOF'
import json
samples = [
    ("User is moving rapidly on foot across the track.", "Running"),
    ("User is sitting perfectly still in a chair.", "Resting"),
    ("User is pedaling a two-wheeled vehicle.", "Cycling"),
    ("User is walking at a leisurely pace.", "Walking"),
    ("User is lifting heavy dumbbells.", "Exercising"),
    ("User is horizontally positioned with eyes closed.", "Sleeping"),
    ("User is moving through water using their arms and legs.", "Swimming"),
    ("User is typing rapidly on a keyboard.", "Working"),
    ("User is chopping vegetables in the kitchen.", "Cooking"),
    ("User is steering a four-wheeled vehicle on the highway.", "Driving"),
]
with open("examples/llm-agent/dataset/activity_classification.jsonl", "w") as f:
    for q_body, a in samples:
        f.write(json.dumps({"question": f"What activity is the user performing? {q_body}", "answer": a}) + "\n")
EOF
```

---

## 4. Configuration files

All four config files already exist in the repo with working defaults. You normally don't need to touch them — but here's what each one controls.

### 4.1 `singletask_learning_bench/benchmarkingjob.yaml`

Top-level job config. Points at `testenv.yaml` and `test_algorithm.yaml`. Also defines the leaderboard format. The metric list (`rouge1`, `rouge2`, `rougeL`) is set under `selected_dataitem.metrics`.

### 4.2 `singletask_learning_bench/testenv/testenv.yaml`

Defines the dataset paths and which metric files to load. Key fields:

- `dataset.train_data` and `dataset.test_data` — paths to the `.jsonl` files. They currently both point at the same file (memorisation setup); change one for a real eval.
- `metrics[].url` — paths to `rouge.py` for each metric.

### 4.3 `singletask_learning_bench/testalgorithms/test_algorithm.yaml`

Points at `basemodel.py` and the two hyperparameter files (`config.json`, `train_config.json`).

### 4.4 `config/config.json`

Runtime config consumed by `basemodel.py`:

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

- `tokenizer_dir` accepts either a HuggingFace Hub id (current default — auto-downloads to the HF cache) or an absolute path to a local model folder.
- `auth_token`: `null` works for public models. Set to a HF token string if you need a gated one.
- `device`: `"auto"` lets `transformers` pick CPU vs GPU. Force with `"cpu"` if needed.

### 4.5 `config/train_config.json`

Passed directly to `transformers.TrainingArguments`:

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

`half_lora` is read and popped by `basemodel.py` before the rest is forwarded to `TrainingArguments` — keep it here.

---

## 5. How the example works

A single run does the following:

1. **Load.** `BaseModel.__init__` reads both JSON configs and loads `Langboat/bloom-1b4-zh` plus its tokenizer.
2. **Preprocess.** For each `(question, answer)` pair, `_preprocess_sample` builds a prompt of the form `"user: \n<question>\n\nassistant: <answer><eos>"` and masks the prompt tokens out of the loss with `-100`.
3. **Train.** Wraps the model in LoRA (`r=16, lora_alpha=32, lora_dropout=0.05`) and runs the HuggingFace `Trainer` for `num_train_epochs` epochs.
4. **Predict.** For each test question, formats it with the same `"user: ...\n\nassistant: "` template, generates 8 new tokens, strips the prompt, takes the first whitespace-split token.
5. **Score.** `rouge.py` runs `rouge_score.RougeScorer` per sample and averages, then multiplies by 10.

The `"user: ... assistant: "` template is shared between training and inference — without that, the trained model emits `<eos>` immediately at predict time and ROUGE collapses to zero.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: not found testenv config file(...)` | YAML path typo | Make sure paths say `llm-agent`, not `LLM-Agent-Benchmark` |
| `NotImplementedError: not one of train_index/train_data/train_data_info` | Bad key in `testenv.yaml` | Use `train_data` / `test_data`, not `train_url` / `test_url` |
| `AttributeError: 'BaseModel' object has no attribute 'preprocess'` | Custom basemodel missing the lifecycle hook | Keep the empty `preprocess(self, *args, **kwargs)` stub |
| All ROUGE scores 0.0, predictions look like rambling sentences | Train/inference prompt mismatch | `predict()` must wrap input in `"user: \n{text}\n\nassistant: "` |
| All ROUGE scores 0.0, predictions are empty strings | Same as above — model emits `<eos>` first | Same fix |
| `train_loss` stays around 7–8 after several epochs | LoRA is too weak | `lora_alpha` should be much larger than `r`; the defaults here are `r=16, alpha=32` |
| Model download stalls | HF Hub rate limiting | Set `HF_TOKEN` env var; or pre-download to a local folder and point `tokenizer_dir` at it |

---

## 7. Tuning for better scores on your own data

The defaults are tuned for the 10-sample demo dataset where train == test. For real data:

- **Split train/test.** Point `train_data` and `test_data` at different files.
- **More data.** 100+ examples per class is a reasonable floor.
- **Drop epochs.** With more data, 3–5 epochs is usually enough; 20 will overfit hard.
- **Multi-word labels.** If your `answer` values are multi-token, drop the `.split()[0]` truncation in `basemodel.py:predict()` so you keep the whole generation. Otherwise rouge2 stays at 0.
- **Different model.** Swap `tokenizer_dir` in `config.json` for any causal LM on the Hub. For English-only tasks, an instruction-tuned model (e.g. `Qwen/Qwen2.5-0.5B-Instruct`) usually beats `bloom-1b4-zh`.

---

## 8. File layout

```
examples/llm-agent/
├── config/
│   ├── config.json                          # runtime config (paths, model id, device)
│   └── train_config.json                    # TrainingArguments
├── dataset/                                 # gitignored; create locally
│   └── activity_classification.jsonl
└── singletask_learning_bench/
    ├── benchmarkingjob.yaml                 # top-level job spec
    ├── requirements.txt                     # torch / transformers / peft / ...
    ├── README.md                            # this file
    ├── testalgorithms/
    │   ├── basemodel.py                     # __init__ / train / predict / preprocess
    │   └── test_algorithm.yaml
    └── testenv/
        ├── rouge.py                         # rouge1 / rouge2 / rougeL metrics
        └── testenv.yaml                     # dataset + metric config
```
