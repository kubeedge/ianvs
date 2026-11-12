# Copyright 2026 The KubeEdge Authors.
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

"""Module ③, the Lifelong Smoke Harness.

A lifelong-learning smoke run passes on exit code 0 alone, so a round that
trained on nothing, or never ran, still passes. For lifelong units this
module:

1. points the benchmark at the prepared smoke dataset (txt index files);
2. checks the dataset gives every round at least one training and one
   evaluation sample, as core splits it (Dataset._splitting_more_times);
3. after the run, checks that every round left its outputs and that the
   rank file was written.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Mapping, Optional, Tuple

LIFELONG_PARADIGM = "lifelonglearning"
DATASET_KEYS = {
    "train": ("train_index", "train_url"),
    "test": ("test_index", "test_url"),
}
DEFAULT_TRAIN_RATIO = 0.8  # core's default lifelong_learning_data_setting
DEFAULT_ROUNDS = 2  # core's TestEnv default


@dataclass
class CheckOutcome:
    status: str  # "PASS", "FAIL" or "SKIP"
    message: str
    details: List[str] = field(default_factory=list)


@dataclass
class LifelongSettings:
    benchmark: dict
    testenv: dict
    rounds: int
    train_ratio: float
    mode: Optional[str]


def load_lifelong_settings(repo_root: Path, benchmark_file: str) -> Optional[LifelongSettings]:
    """Settings of a lifelong benchmark, or None if it is not lifelong."""
    benchmark = _load_yaml(repo_root / benchmark_file)
    job = benchmark.get("benchmarkingjob") if isinstance(benchmark, dict) else None
    if not isinstance(job, dict):
        return None
    algorithms = (job.get("test_object") or {}).get("algorithms") or []
    algorithm = {}
    for entry in algorithms:
        url = entry.get("url") if isinstance(entry, dict) else None
        data = _load_yaml(repo_root / _repo_path(url)) if url else {}
        candidate = data.get("algorithm") if isinstance(data, dict) else None
        if isinstance(candidate, dict) and \
                str(candidate.get("paradigm_type", "")).lower() == LIFELONG_PARADIGM:
            algorithm = candidate
            break
    if not algorithm:
        return None

    testenv = _load_yaml(repo_root / _repo_path(str(job.get("testenv", ""))))
    testenv_body = testenv.get("testenv", {}) if isinstance(testenv, dict) else {}
    setting = algorithm.get("lifelong_learning_data_setting") or {}
    model_metric = (testenv_body.get("model_eval") or {}).get("model_metric") or {}
    return LifelongSettings(
        benchmark=benchmark,
        testenv=testenv,
        rounds=int(testenv_body.get("incremental_rounds", DEFAULT_ROUNDS)),
        train_ratio=float(setting.get("train_ratio", DEFAULT_TRAIN_RATIO)),
        mode=model_metric.get("mode"),
    )


def min_samples_per_round(train_ratio: float) -> int:
    """Smallest slice that core splits into >= 1 training and >= 1 eval sample."""
    size = 1
    while int(size * train_ratio) < 1 or size - int(size * train_ratio) < 1:
        size += 1
        if size > 10000:
            break
    return size


def check_sizing(train_index: Path, settings: LifelongSettings) -> CheckOutcome:
    if not train_index.is_file():
        return CheckOutcome("FAIL", f"Smoke training index is missing: {train_index.name}.")
    samples = [line for line in train_index.read_text(encoding="utf-8").splitlines() if line.strip()]
    rounds = settings.rounds
    per_round = int(len(samples) / rounds) if rounds else 0
    needed = min_samples_per_round(settings.train_ratio)
    message = (
        f"{len(samples)} training samples over {rounds} rounds gives {per_round} per round; "
        f"train_ratio {settings.train_ratio} needs at least {needed} per round "
        f"(N >= {rounds * needed})."
    )
    if per_round >= needed:
        return CheckOutcome("PASS", message)
    return CheckOutcome(
        "FAIL",
        message + " Impact: some round would train or evaluate on no samples, "
                  "and nothing in the run would report it.",
    )


def materialize_benchmark(
    settings: LifelongSettings,
    smoke_root: Path,
    temp_dir: Path,
) -> Tuple[str, Path, Path]:
    """Write a benchmark that uses the smoke dataset and a temporary workspace.

    Returns (benchmark file, training index, workspace). The testenv keeps its
    own dataset keys (legacy or modern); only their values change.
    """
    import yaml

    testenv = dict(settings.testenv)
    body = dict(testenv.get("testenv", {}))
    dataset = dict(body.get("dataset") or {})
    indexes = {}
    for split, keys in DATASET_KEYS.items():
        index = smoke_root / f"{split}_data" / "index.txt"
        indexes[split] = index
        present = [key for key in keys if key in dataset]
        for key in present or keys[:1]:
            dataset[key] = str(index)
    body["dataset"] = dataset
    testenv["testenv"] = body

    benchmark = dict(settings.benchmark)
    job = dict(benchmark["benchmarkingjob"])
    workspace = temp_dir / "workspace"
    testenv_file = temp_dir / "testenv.yaml"
    job["testenv"] = str(testenv_file)
    job["workspace"] = str(workspace)
    benchmark["benchmarkingjob"] = job

    testenv_file.write_text(yaml.safe_dump(testenv, sort_keys=False), encoding="utf-8")
    benchmark_file = temp_dir / "benchmarkingjob.yaml"
    benchmark_file.write_text(yaml.safe_dump(benchmark, sort_keys=False), encoding="utf-8")
    return str(benchmark_file), indexes["train"], workspace


def check_round_completion(workspace: Path, settings: LifelongSettings) -> CheckOutcome:
    """Every round left its outputs, and the rank file exists."""
    if settings.mode is not None:
        return CheckOutcome(
            "SKIP",
            f"Round outputs are only checked for the default mode; this benchmark "
            f"uses mode `{settings.mode}`.",
        )
    outputs = sorted(path.parent for path in workspace.glob("*/*/*/output/train"))
    if not outputs:
        return CheckOutcome("FAIL", "No lifelong-learning output directory was written.")

    missing = []
    for output in outputs:
        missing.extend(_missing_round_outputs(output, settings.rounds))
    ranks = list(workspace.glob("*/rank/all_rank.csv"))
    if not ranks or not any(len(_csv_rows(rank)) > 1 for rank in ranks):
        missing.append("rank/all_rank.csv is missing or has no result rows")

    if missing:
        return CheckOutcome(
            "FAIL",
            "The run exited, but not every round completed. Impact: a skipped or empty "
            "round would otherwise pass on exit code alone.",
            missing,
        )
    return CheckOutcome(
        "PASS",
        f"All {settings.rounds} rounds wrote their outputs and the rank file exists.",
    )


def _missing_round_outputs(output: Path, rounds: int) -> List[str]:
    """Default mode: round 1 trains and evaluates; later rounds infer, then
    train and evaluate only when unseen samples were found."""
    missing = []
    for name in ("train/1", "eval/1"):
        if not (output / name).is_dir():
            missing.append(f"round 1: output/{name} is missing")
    for round_number in range(2, rounds + 1):
        if not (output / "inference" / "results" / str(round_number)).is_dir():
            missing.append(f"round {round_number}: output/inference/results/{round_number} is missing")
            continue
        unseen = output / "inference" / "unseen_task_samples" / str(round_number)
        if unseen.is_dir() and any(unseen.iterdir()):
            for name in (f"train/{round_number}", f"eval/{round_number}"):
                if not (output / name).is_dir():
                    missing.append(f"round {round_number}: output/{name} is missing")
    return missing


def _csv_rows(path: Path) -> List[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_yaml(path: Path) -> Mapping:
    import yaml

    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def _repo_path(value: str) -> str:
    value = value.strip().strip("\"'")
    return value[2:] if value.startswith("./") else value
