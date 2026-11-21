# Copyright 2024 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Prepare the GPQA-Diamond dataset for the cloud-edge-collaborative-inference-for-llm
example, without requiring a Kaggle account or a gated HuggingFace login.

Background
----------
The canonical GPQA dataset (Idavidrein/gpqa on HuggingFace) is *gated*: users
must log in and accept terms of use before downloading it. The README of this
example previously pointed users at a Kaggle mirror for MMLU-5-shot only, and
implied "the same progress" works for GPQA-diamond -- but never actually gave
a working GPQA download link, and the Kaggle route additionally requires a
kaggle.json auth token that is a common source of setup failures
(see issues #332, #330, #357 on kubeedge/ianvs).

This script instead pulls GPQA-diamond from the public, unauthenticated CSV
mirror that OpenAI's own `simple-evals` benchmarking tool uses:

    https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv

No login, no token, no account needed -- just a plain HTTPS GET.

Usage
-----
    python prepare_gpqa_dataset.py [--output-dir ./dataset/gpqa]

This produces:

    dataset/gpqa/
    ├── train_data
    │   └── data.json         # empty, per Ianvs convention for this example
    └── test_data
        ├── data.jsonl        # one JSON object per question
        └── metadata.json     # dataset-level metadata

which matches the paths already expected by
examples/cloud-edge-collaborative-inference-for-llm/testenv/testenv.yaml.
"""

import argparse
import csv
import io
import json
import random
import sys
import urllib.request
from pathlib import Path

GPQA_DIAMOND_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"
)

# Fixed seed so that the A/B/C/D shuffling of answer choices is reproducible
# across machines and across repeated runs -- required for the benchmark's
# results to be comparable, per the example's own reproducibility goals.
SHUFFLE_SEED = 0


def download_csv(url: str) -> str:
    """Download the CSV as text, raising a clear error on failure."""
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"Unexpected HTTP status {response.status} while "
                    f"downloading {url}"
                )
            return response.read().decode("utf-8")
    except Exception as exc:  # noqa: BLE001 -- surface a helpful message
        raise RuntimeError(
            f"Failed to download GPQA-diamond dataset from {url}.\n"
            "This URL is the same public mirror used by OpenAI's "
            "simple-evals project and should not require authentication. "
            "If this keeps failing, check your network connection or "
            "see https://github.com/openai/simple-evals for the canonical "
            "source."
        ) from exc


def rows_to_ianvs_examples(csv_text: str, rng: random.Random):
    """Convert simple-evals' GPQA CSV rows into Ianvs' query/response format.

    The source CSV has one row per question with columns:
        Question, Correct Answer,
        Incorrect Answer 1, Incorrect Answer 2, Incorrect Answer 3
    (plus some additional metadata columns depending on the export).

    Ianvs expects each line of test_data/data.jsonl to be a JSON object with
    keys: query, response, explanation, level_1_dim .. level_4_dim -- see the
    MMLU-5-shot example already documented in this example's README.
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    examples = []

    for row in reader:
        question = row.get("Question", "").strip()
        correct = row.get("Correct Answer", "").strip()
        incorrect = [
            row.get("Incorrect Answer 1", "").strip(),
            row.get("Incorrect Answer 2", "").strip(),
            row.get("Incorrect Answer 3", "").strip(),
        ]

        if not question or not correct or any(a == "" for a in incorrect):
            # Skip malformed rows rather than silently corrupting the
            # benchmark -- but keep going so one bad row doesn't block the
            # whole dataset.
            continue

        choices = incorrect + [correct]
        rng.shuffle(choices)
        correct_index = choices.index(correct)
        correct_letter = "ABCD"[correct_index]

        formatted_query = (
            f"Question: {question}\n"
            f"A. {choices[0]}\n"
            f"B. {choices[1]}\n"
            f"C. {choices[2]}\n"
            f"D. {choices[3]}"
        )

        examples.append(
            {
                "query": formatted_query,
                "response": correct_letter,
                "explanation": "",
                "level_1_dim": "single-modal",
                "level_2_dim": "text",
                "level_3_dim": "knowledge Q&A",
                "level_4_dim": "gpqa-diamond",
            }
        )

    if not examples:
        raise RuntimeError(
            "Parsed 0 valid examples from the GPQA-diamond CSV. The source "
            "CSV's column layout may have changed -- inspect the raw CSV "
            "and update rows_to_ianvs_examples() accordingly."
        )

    return examples


def write_dataset(examples, output_dir: Path):
    train_dir = output_dir / "train_data"
    test_dir = output_dir / "test_data"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Per the existing MMLU-5-shot example convention, train_data/data.json
    # is left empty for this benchmark (no fine-tuning split is used).
    (train_dir / "data.json").write_text("", encoding="utf-8")

    data_jsonl_path = test_dir / "data.jsonl"
    with data_jsonl_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    metadata = {
        "dataset": "GPQA-Diamond",
        "description": (
            "GPQA is a multiple-choice Q&A dataset of graduate-level "
            "questions in biology, physics, and chemistry, written and "
            "validated by domain experts (Rein et al., 2023). This is the "
            "Diamond subset (highest-quality, most-agreed-upon questions), "
            "sourced from the public mirror used by OpenAI's simple-evals "
            "(no authentication required)."
        ),
        "source": GPQA_DIAMOND_URL,
        "license": "CC BY 4.0 (see https://github.com/idavidrein/gpqa)",
        "level_1_dim": "single-modal",
        "level_2_dim": "text",
        "level_3_dim": "knowledge Q&A",
        "level_4_dim": "gpqa-diamond",
    }
    (test_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return data_jsonl_path, test_dir / "metadata.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./dataset/gpqa"),
        help=(
            "Where to write the Ianvs-formatted dataset. Defaults to "
            "./dataset/gpqa, matching testenv.yaml's default paths."
        ),
    )
    args = parser.parse_args()

    print(f"Downloading GPQA-diamond from {GPQA_DIAMOND_URL} ...")
    csv_text = download_csv(GPQA_DIAMOND_URL)

    print("Formatting examples into Ianvs schema ...")
    rng = random.Random(SHUFFLE_SEED)
    examples = rows_to_ianvs_examples(csv_text, rng)
    print(f"Parsed {len(examples)} valid questions.")

    data_path, metadata_path = write_dataset(examples, args.output_dir)
    print(f"Wrote {data_path}")
    print(f"Wrote {metadata_path}")
    print(
        "\nDone. No Kaggle account or HuggingFace login was required.\n"
        f"Dataset root: {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
