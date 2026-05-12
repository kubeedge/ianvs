#!/usr/bin/env python3
# Copyright 2022 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Prepare the GovAff Kaggle dataset for this Ianvs example."""

import argparse
import json
import shutil
from pathlib import Path


OBJECTIVE_METADATA = {
    "dataset": "A Objective BenchMark Template",
    "description": "A government benchmark for llm testing",
    "level_1_dim": "single-modal",
    "level_2_dim": "text",
    "level_3_dim": "Q&A",
    "level_4_dim": "government",
}

SUBJECTIVE_METADATA = {
    "dataset": "A Subjective BenchMark Template",
    "description": "A government benchmark for llm testing",
    "level_1_dim": "single-modal",
    "level_2_dim": "text",
    "level_3_dim": "Q&A",
    "level_4_dim": "government",
}


def _read_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=4)
        file.write("\n")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _format_options(options):
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "\n".join(f"{labels[index]}. {option}" for index, option in enumerate(options))


def _prepare_objective(source_dir, output_dir):
    source_data = source_dir / "multi-choice questions" / "data.jsonl"
    if not source_data.is_file():
        raise FileNotFoundError(f"missing objective source data: {source_data}")

    train_rows = []
    test_rows = []
    for row in _read_jsonl(source_data):
        question = row["question"]
        answer = row["answer"]
        train_rows.append({"question": question, "answer": answer})
        test_rows.append({
            "query": f"{question}\n{_format_options(row['options'])}",
            "response": answer,
            "level_1_dim": OBJECTIVE_METADATA["level_1_dim"],
            "level_2_dim": OBJECTIVE_METADATA["level_2_dim"],
            "level_3_dim": OBJECTIVE_METADATA["level_3_dim"],
            "level_4_dim": OBJECTIVE_METADATA["level_4_dim"],
        })

    target = output_dir / "objective"
    _write_jsonl(target / "train_data" / "data.jsonl", train_rows)
    _write_jsonl(target / "test_data" / "data.jsonl", test_rows)
    _write_json(target / "test_data" / "metadata.json", OBJECTIVE_METADATA)


def _prepare_subjective(source_dir, output_dir):
    source_base = source_dir / "subjective questions"
    source_data = source_base / "data.jsonl"
    source_metadata = source_base / "metadata.json"
    if not source_data.is_file():
        raise FileNotFoundError(f"missing subjective source data: {source_data}")
    if not source_metadata.is_file():
        raise FileNotFoundError(f"missing subjective metadata: {source_metadata}")

    train_rows = [
        {"question": row["query"], "answer": row["response"]}
        for row in _read_jsonl(source_data)
    ]

    target = output_dir / "subjective"
    _write_jsonl(target / "train_data" / "data.jsonl", train_rows)
    (target / "test_data").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_data, target / "test_data" / "data.jsonl")
    shutil.copyfile(source_metadata, target / "test_data" / "metadata.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("dataset/govaff_raw/government"),
        help="Directory containing the unzipped Kaggle 'government' folder.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/government"),
        help="Directory where Ianvs-ready objective/subjective data is written.",
    )
    args = parser.parse_args()

    source_dir = args.source.resolve()
    output_dir = args.output.resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {source_dir}")

    _prepare_objective(source_dir, output_dir)
    _prepare_subjective(source_dir, output_dir)
    print(f"Prepared government benchmark dataset under {output_dir}")


if __name__ == "__main__":
    main()
