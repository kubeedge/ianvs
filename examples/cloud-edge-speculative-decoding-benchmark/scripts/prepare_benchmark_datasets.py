#!/usr/bin/env python3
import argparse
import gzip
import json
import re
import sys
import urllib.request
from pathlib import Path


GSM8K_DATASET = ("openai/gsm8k", "main")
HUMANEVAL_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
MT_BENCH_URL = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"

METADATA = {
    "gsm8k": {
        "dataset": "gsm8k",
        "description": "gsm8k single-turn benchmark dataset in Ianvs standard test_data layout.",
        "level_1_dim": "single-modal",
        "level_2_dim": "text",
        "level_3_dim": "task-category",
        "level_4_dim": "benchmark-name",
    },
    "humaneval": {
        "dataset": "humaneval",
        "description": "humaneval code-completion benchmark dataset in Ianvs standard test_data layout.",
        "level_1_dim": "single-modal",
        "level_2_dim": "text",
        "level_3_dim": "task-category",
        "level_4_dim": "benchmark-name",
    },
    "mt_bench": {
        "dataset": "mt_bench",
        "description": "mt_bench multi-turn benchmark dataset converted to Ianvs standard test_data layout.",
        "level_1_dim": "single-modal",
        "level_2_dim": "text",
        "level_3_dim": "task-category",
        "level_4_dim": "benchmark-name",
    },
}


def _require_datasets():
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'datasets'. Install example requirements first: "
            "pip install -r examples/cloud-edge-speculative-decoding-benchmark/requirements.txt"
        ) from exc
    return load_dataset


def _download_text_lines(url: str, gzipped: bool = False) -> list[str]:
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read()
    if gzipped:
        payload = gzip.decompress(payload)
    return payload.decode("utf-8").splitlines()


def _extract_question_id(value, default: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = re.search(r"(\d+)$", value)
        if match:
            return int(match.group(1))
    return default


def _ensure_layout(dst_root: Path, dataset_name: str):
    dataset_dir = dst_root / dataset_name
    test_dir = dataset_dir / "test_data"
    train_dir = dataset_dir / "train_data"
    test_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    return test_dir / "data.jsonl", test_dir / "metadata.json", train_dir / "data.jsonl"


def _write_layout(dst_root: Path, dataset_name: str, rows: list[dict]) -> dict:
    test_path, meta_path, train_path = _ensure_layout(dst_root, dataset_name)
    with test_path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta_path.write_text(json.dumps(METADATA[dataset_name], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    train_path.write_text("", encoding="utf-8")
    return {"dataset": dataset_name, "target": str(test_path), "count": len(rows)}


def build_gsm8k_rows() -> list[dict]:
    load_dataset = _require_datasets()
    dataset = load_dataset(GSM8K_DATASET[0], GSM8K_DATASET[1], split="test")
    rows = []
    for index, item in enumerate(dataset):
        rows.append(
            {
                "request_id": f"gsm8k-{index:03d}",
                "query": item["question"],
                "response": item["answer"],
                "task_name": "gsm8k",
                "level_3_dim": "math",
                "level_4_dim": "gsm8k",
                "question_id": index,
                "category": "math",
            }
        )
    return rows


def build_humaneval_rows() -> list[dict]:
    lines = _download_text_lines(HUMANEVAL_URL, gzipped=True)
    rows = []
    for index, line in enumerate(lines):
        item = json.loads(line)
        question_id = _extract_question_id(item.get("task_id"), index)
        rows.append(
            {
                "request_id": f"humaneval-{index:03d}",
                "query": "Complete the code I provided.\n\n" + item["prompt"],
                "response": item["canonical_solution"],
                "task_name": "humaneval",
                "level_3_dim": "code",
                "level_4_dim": "humaneval",
                "question_id": question_id,
                "category": "code",
            }
        )
    return rows


def build_mt_bench_rows() -> list[dict]:
    lines = _download_text_lines(MT_BENCH_URL, gzipped=False)
    rows = []
    for index, line in enumerate(lines):
        item = json.loads(line)
        turns = item.get("turns") or []
        question_id = _extract_question_id(item.get("question_id"), index)
        category = item.get("category") or "mt_bench"
        conversation_id = f"mt_bench-{question_id:03d}"
        for turn_index, turn in enumerate(turns):
            request_id = f"{conversation_id}-turn-{turn_index}"
            payload = {
                "request_id": request_id,
                "query": str(turn),
                "task_name": "mt_bench",
                "conversation_id": conversation_id,
                "turn_index": turn_index,
                "turn_count": len(turns),
                "question_id": question_id,
                "category": category,
            }
            rows.append(
                {
                    "request_id": request_id,
                    "query": json.dumps(payload, ensure_ascii=False),
                    "response": "",
                    "task_name": "mt_bench",
                    "level_3_dim": category,
                    "level_4_dim": "mt_bench",
                    "question_id": question_id,
                    "category": category,
                    "conversation_id": conversation_id,
                    "turn_index": turn_index,
                    "turn_count": len(turns),
                }
            )
    return rows


BUILDERS = {
    "gsm8k": build_gsm8k_rows,
    "humaneval": build_humaneval_rows,
    "mt_bench": build_mt_bench_rows,
}


def main():
    parser = argparse.ArgumentParser(
        description="Download public benchmark datasets and convert them into Ianvs example layout."
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        default=Path("./examples/cloud-edge-speculative-decoding-benchmark/dataset"),
        help="Destination dataset root in Ianvs example layout",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["gsm8k", "humaneval", "mt_bench"],
        choices=["gsm8k", "humaneval", "mt_bench"],
        help="Datasets to prepare",
    )
    args = parser.parse_args()

    summary = []
    for dataset_name in args.datasets:
        rows = BUILDERS[dataset_name]()
        record = _write_layout(args.dst_root, dataset_name, rows)
        summary.append(record)
    json.dump(summary, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
