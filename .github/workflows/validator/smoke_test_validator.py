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

"""Environment preparation, smoke, and JSONL validation for Ianvs examples."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from services.inventory_loader import DEFAULT_INVENTORY_PATH, load_inventory_examples
from static_validator import (
    FAIL,
    PASS,
    SKIP,
    CheckResult,
    ExampleReport,
    StaticValidationReport,
    render_json,
    render_markdown,
)


DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_DATASET_ROOT = "dataset"
PREPARATION_STEP_FIELDS = ("name", "type", "script", "args", "timeout")


@dataclass(frozen=True)
class DatasetConfig:
    paths: List[str]
    root: str


def validate_examples(
    repo_root: Path,
    examples: Sequence[Mapping[str, object]],
    execute: bool = True,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    smoke_command: Sequence[str] = (),
) -> StaticValidationReport:
    reports = [
        validate_example(
            repo_root=repo_root,
            example=example,
            execute=execute,
            timeout_seconds=timeout_seconds,
            smoke_command=smoke_command,
        )
        for example in examples
    ]
    return StaticValidationReport(reports=reports)


def validate_jsonl_examples(
    repo_root: Path,
    examples: Sequence[Mapping[str, object]],
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> StaticValidationReport:
    reports = []
    for example in examples:
        example_path = _normalize_repo_path(str(example.get("path", "")))
        report = ExampleReport(
            name=str(example.get("name") or example_path),
            path=example_path,
        )
        with tempfile.TemporaryDirectory(prefix="ianvs-jsonl-") as temp_dir:
            dataset_config = _dataset_config_from_example(repo_root, example)
            temp_dataset_root = _temporary_dataset_root(temp_dir, dataset_config, example)
            prepared_dataset_root = _prepare_dataset(
                report,
                repo_root,
                example,
                temp_dataset_root,
                timeout_seconds,
            )
            dataset_files = _dataset_paths_from_config(
                repo_root,
                example,
                prepared_dataset_root=prepared_dataset_root,
            )
            _check_jsonl_dataset(report, repo_root, dataset_files)
        reports.append(report)
    return StaticValidationReport(reports=reports)


def prepare_example_environments(
    repo_root: Path,
    examples: Sequence[Mapping[str, object]],
) -> StaticValidationReport:
    """Execute the ordered prepare_env steps declared by each example."""
    return StaticValidationReport(
        reports=[_prepare_example_environment(repo_root, example) for example in examples]
    )


def _prepare_example_environment(
    repo_root: Path,
    example: Mapping[str, object],
) -> ExampleReport:
    example_path = _normalize_repo_path(str(example.get("path", "")))
    report = ExampleReport(
        name=str(example.get("name") or example_path),
        path=example_path,
    )
    config = example.get("prepare_env")
    if not isinstance(config, Mapping):
        _append_check(
            report,
            name="Environment preparation",
            status=SKIP,
            message="No prepare_env configuration is declared.",
        )
        return report

    working_directory = config.get("working_directory")
    steps = config.get("steps")
    if not isinstance(working_directory, str) or not working_directory.strip():
        _append_preparation_configuration_failure(
            report,
            "prepare_env.working_directory must be a string.",
        )
        return report
    if not isinstance(steps, list) or not steps:
        _append_preparation_configuration_failure(
            report,
            "prepare_env.steps must be a non-empty array.",
        )
        return report

    try:
        workdir = _resolve_within_repo(repo_root, working_directory)
    except ValueError as error:
        _append_preparation_configuration_failure(report, str(error))
        return report
    if not workdir.is_dir():
        _append_preparation_configuration_failure(
            report,
            "Working directory does not exist: {}".format(working_directory),
        )
        return report

    for index, step in enumerate(steps, 1):
        issue = _validate_preparation_step(step, index)
        if issue:
            _append_preparation_configuration_failure(report, issue)
            break
        assert isinstance(step, Mapping)
        if not _run_preparation_step(report, repo_root, workdir, step):
            break
    return report


def _validate_preparation_step(step: object, index: int) -> str:
    prefix = "prepare_env.steps[{}]".format(index - 1)
    if not isinstance(step, Mapping):
        return "{} must be an object.".format(prefix)
    missing = [field for field in PREPARATION_STEP_FIELDS if field not in step]
    if missing:
        return "{} is missing required fields: {}.".format(
            prefix,
            ", ".join(missing),
        )
    for field in ("name", "type", "script"):
        if not isinstance(step[field], str) or not str(step[field]).strip():
            return "{}.{} must be a non-empty string.".format(prefix, field)
    args = step["args"]
    if not isinstance(args, list) or any(not isinstance(arg, str) for arg in args):
        return "{}.args must be an array of strings.".format(prefix)
    timeout = step["timeout"]
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        return "{}.timeout must be a positive integer.".format(prefix)
    return ""


def _run_preparation_step(
    report: ExampleReport,
    repo_root: Path,
    workdir: Path,
    step: Mapping[str, object],
) -> bool:
    name = str(step["name"])
    step_type = str(step["type"])
    script = str(step["script"])
    args = list(step["args"])
    timeout = int(step["timeout"])
    try:
        script_path = _resolve_within_repo(workdir, script, root=repo_root)
    except ValueError as error:
        _append_preparation_step_result(
            report,
            name,
            step_type,
            script,
            FAIL,
            str(error),
        )
        return False
    if not script_path.is_file():
        _append_preparation_step_result(
            report,
            name,
            step_type,
            script,
            FAIL,
            "Preparation script does not exist.",
        )
        return False

    command: List[str]
    if script_path.suffix == ".py":
        command = [sys.executable, str(script_path), *args]
    else:
        command = [str(script_path), *args]

    try:
        completed = subprocess.run(
            command,
            cwd=str(workdir),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        _append_preparation_step_result(
            report,
            name,
            step_type,
            script,
            FAIL,
            "Preparation step timed out after {} seconds.".format(timeout),
            _summarize_output(output),
        )
        return False
    except OSError as error:
        _append_preparation_step_result(
            report,
            name,
            step_type,
            script,
            FAIL,
            "Could not start preparation step: {}".format(error),
        )
        return False

    passed = completed.returncode == 0
    _append_preparation_step_result(
        report,
        name,
        step_type,
        script,
        PASS if passed else FAIL,
        (
            "Preparation step completed successfully."
            if passed
            else "Preparation step failed with exit code {}.".format(
                completed.returncode
            )
        ),
        _summarize_output(completed.stdout),
    )
    return passed


def _resolve_within_repo(
    base: Path,
    value: str,
    root: Optional[Path] = None,
) -> Path:
    repository = (root or base).resolve()
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = base / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(repository)
    except ValueError:
        raise ValueError("Path escapes the repository: {}".format(value))
    return candidate


def _append_preparation_configuration_failure(
    report: ExampleReport,
    message: str,
) -> None:
    _append_check(
        report,
        name="Environment preparation configuration",
        status=FAIL,
        message=message,
    )


def _append_preparation_step_result(
    report: ExampleReport,
    name: str,
    step_type: str,
    script: str,
    status: str,
    message: str,
    details: Sequence[str] = (),
) -> None:
    _append_check(
        report,
        name="Environment preparation: {} ({})".format(name, step_type),
        status=status,
        file=script,
        message=message,
        details=details,
    )


def validate_example(
    repo_root: Path,
    example: Mapping[str, object],
    execute: bool,
    timeout_seconds: int,
    smoke_command: Sequence[str],
) -> ExampleReport:
    example_path = _normalize_repo_path(str(example.get("path", "")))
    report = ExampleReport(
        name=str(example.get("name") or example_path),
        path=example_path,
    )
    benchmark_file = _example_file(
        example,
        "benchmark_file",
        example_path,
        "benchmarkingjob.yaml",
    )
    benchmark_path = repo_root / benchmark_file
    _append_check(
        report,
        name="Smoke benchmark config exists",
        status=PASS if benchmark_path.is_file() else FAIL,
        file=benchmark_file,
        message=(
            "Benchmark configuration exists."
            if benchmark_path.is_file()
            else "Benchmark configuration is missing."
        ),
    )
    if not benchmark_path.is_file():
        return report

    with tempfile.TemporaryDirectory(prefix="ianvs-smoke-") as temp_dir:
        dataset_config = _dataset_config_from_example(repo_root, example)
        temp_dataset_root = _temporary_dataset_root(temp_dir, dataset_config, example)
        prepared_dataset_root = None
        if not isinstance(example.get("prepare_env"), Mapping):
            # Backward compatibility for inventory entries that still use the
            # legacy dataset.prepare_script contract.
            prepared_dataset_root = _prepare_dataset(
                report,
                repo_root,
                example,
                temp_dataset_root,
                timeout_seconds,
            )
        dataset_files = _dataset_paths_from_config(
            repo_root,
            example,
            prepared_dataset_root=prepared_dataset_root,
        )
        _check_jsonl_dataset(report, repo_root, dataset_files)
        command_benchmark_file = _materialize_smoke_benchmark(
            repo_root,
            benchmark_file,
            prepared_dataset_root,
            Path(temp_dir),
        )

        if execute:
            _run_smoke_command(
                report,
                repo_root,
                example,
                command_benchmark_file,
                prepared_dataset_root,
                smoke_command,
                timeout_seconds,
            )
        else:
            _append_check(
                report,
                name="Runtime smoke test",
                status=SKIP,
                file=benchmark_file,
                message="Runtime execution was not requested.",
            )

    return report


def render_smoke_markdown(report: StaticValidationReport) -> str:
    rendered = render_markdown(report)
    return rendered.replace("# Static Validation Report", "# Smoke Validation Report", 1)


def _prepare_dataset(
    report: ExampleReport,
    repo_root: Path,
    example: Mapping[str, object],
    dataset_root: Path,
    timeout_seconds: int,
) -> Optional[Path]:
    prepare_script = _nested_file(example, ("dataset", "prepare_script"))
    if not prepare_script:
        _append_check(
            report,
            name="Dataset preparation",
            status=SKIP,
            message="No dataset preparation script is declared.",
        )
        return None

    script_path = repo_root / prepare_script
    if not script_path.is_file():
        _append_check(
            report,
            name="Dataset preparation",
            status=FAIL,
            file=prepare_script,
            message="Declared dataset preparation script is missing.",
        )
        return None

    command = [sys.executable, prepare_script]
    script_text = script_path.read_text(encoding="utf-8")
    if "--dataset-root" in script_text:
        command.extend(["--dataset-root", str(dataset_root)])
    if "--smoke" in script_text:
        command.append("--smoke")

    try:
        completed = subprocess.run(
            command,
            cwd=str(repo_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        _append_check(
            report,
            name="Dataset preparation",
            status=FAIL,
            file=prepare_script,
            message="Dataset preparation timed out after {} seconds.".format(
                timeout_seconds
            ),
        )
        return None

    _append_check(
        report,
        name="Dataset preparation",
        status=PASS if completed.returncode == 0 else FAIL,
        file=prepare_script,
        message=(
            "Dataset preparation completed successfully."
            if completed.returncode == 0
            else "Dataset preparation failed with exit code {}.".format(
                completed.returncode
            )
        ),
        details=_summarize_output(completed.stdout),
    )
    if completed.returncode != 0:
        return None
    return dataset_root


def _dataset_paths_from_config(
    repo_root: Path,
    example: Mapping[str, object],
    prepared_dataset_root: Optional[Path] = None,
) -> List[Path]:
    dataset_config = _dataset_config_from_example(repo_root, example)

    paths = []
    for value in dataset_config.paths:
        normalized = _normalize_repo_path(value)
        if (
            prepared_dataset_root
            and dataset_config.root
            and _path_is_within(normalized, dataset_config.root)
        ):
            relative = _relative_dataset_path(normalized, dataset_config.root)
            paths.append(prepared_dataset_root / relative)
        else:
            paths.append(repo_root / normalized)
    return paths


def _dataset_config_from_example(
    repo_root: Path,
    example: Mapping[str, object],
) -> DatasetConfig:
    paths = _dataset_paths_from_inventory(example)
    if not paths:
        paths = _dataset_paths_from_testenv(repo_root, example)
    root = _dataset_root_from_inventory(example) or _dataset_root_from_paths(paths)
    return DatasetConfig(paths=paths, root=root)


def _dataset_paths_from_inventory(example: Mapping[str, object]) -> List[str]:
    dataset = example.get("dataset")
    if not isinstance(dataset, Mapping):
        return []
    root = dataset.get("root")
    structure = dataset.get("structure")
    if not root or not isinstance(structure, list):
        return []
    return [
        _join_repo_path(str(root), str(item))
        for item in structure
        if str(item).endswith(".jsonl")
    ]


def _dataset_root_from_inventory(example: Mapping[str, object]) -> str:
    dataset = example.get("dataset")
    if not isinstance(dataset, Mapping):
        return ""
    root = dataset.get("root")
    if not root:
        return ""
    return _normalize_repo_path(str(root))


def _dataset_paths_from_testenv(
    repo_root: Path,
    example: Mapping[str, object],
) -> List[str]:
    benchmark_file = _optional_example_file(example, "benchmark_file")
    if not benchmark_file:
        return []
    try:
        import yaml
    except ImportError:
        return []

    benchmark_path = repo_root / benchmark_file
    if not benchmark_path.is_file():
        return []
    benchmark = yaml.safe_load(benchmark_path.read_text(encoding="utf-8")) or {}
    testenv_path = (
        benchmark.get("benchmarkingjob", {})
        .get("testenv", "")
    )
    if not testenv_path:
        return []

    testenv_file = repo_root / _normalize_repo_path(str(testenv_path))
    if not testenv_file.is_file():
        return []
    testenv = yaml.safe_load(testenv_file.read_text(encoding="utf-8")) or {}
    return _dataset_config_from_testenv_payload(testenv).paths


def _dataset_root_from_paths(paths: Sequence[str]) -> str:
    if not paths:
        return ""
    parent_paths = [
        _normalize_repo_path(path).rstrip("/").rsplit("/", 1)[0]
        for path in paths
        if "/" in _normalize_repo_path(path).rstrip("/")
    ]
    if not parent_paths:
        return ""
    if len(parent_paths) == 1:
        return parent_paths[0]
    common_root = os.path.commonpath(parent_paths)
    return _normalize_repo_path(common_root)


def _temporary_dataset_root(
    temp_dir: str,
    dataset_config: DatasetConfig,
    example: Mapping[str, object],
) -> Path:
    root = dataset_config.root or _join_repo_path(
        DEFAULT_DATASET_ROOT,
        _safe_path_component(str(example.get("name") or example.get("path") or "example")),
    )
    if Path(root).is_absolute():
        root = _join_repo_path(DEFAULT_DATASET_ROOT, _safe_path_component(root))
    return Path(temp_dir) / root


def _check_jsonl_dataset(
    report: ExampleReport,
    repo_root: Path,
    dataset_paths: Sequence[Path],
) -> None:
    if not dataset_paths:
        _append_check(
            report,
            name="JSONL dataset structure",
            status=SKIP,
            message="No JSONL dataset paths were discovered.",
        )
        return

    issues = []
    for path in dataset_paths:
        allow_empty = _looks_like_training_data_path(path)
        issues.extend(_validate_jsonl_file(path, repo_root, allow_empty=allow_empty))

    _append_issue_check(
        report,
        name="JSONL dataset structure",
        issues=issues,
        fail_message="JSONL dataset validation failed.",
        pass_message="JSONL dataset files exist and contain valid JSON object rows.",
    )


def _validate_jsonl_file(
    path: Path,
    repo_root: Path,
    allow_empty: bool = False,
) -> List[str]:
    display_path = _display_path(path, repo_root)
    if not path.is_file():
        return ["{}: file is missing".format(display_path)]

    rows = path.read_text(encoding="utf-8").splitlines()
    if not rows:
        if allow_empty:
            return []
        return ["{}: file is empty".format(display_path)]

    issues = []
    for line_number, line in enumerate(rows, 1):
        if not line.strip():
            issues.append("{}:{}: blank line".format(display_path, line_number))
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append("{}:{}: {}".format(display_path, line_number, exc))
            continue
        if not isinstance(payload, dict):
            issues.append("{}:{}: row is not a JSON object".format(display_path, line_number))
            continue
    return issues


def _run_smoke_command(
    report: ExampleReport,
    repo_root: Path,
    example: Mapping[str, object],
    benchmark_file: str,
    prepared_dataset_root: Optional[Path],
    smoke_command: Sequence[str],
    timeout_seconds: int,
) -> None:
    command = list(smoke_command) if smoke_command else _default_smoke_command(benchmark_file)
    env = dict(os.environ)
    if prepared_dataset_root:
        env["IANVS_SMOKE_DATASET_ROOT"] = str(prepared_dataset_root)
    mocked_llm, mock_error = _configure_mock_runtime(env, repo_root, example)
    check_name = "Runtime smoke test (mocked_llm)" if mocked_llm else "Runtime smoke test"
    if mock_error:
        _append_check(
            report,
            name="Runtime smoke test (mocked_llm)",
            status=FAIL,
            file=benchmark_file,
            message=mock_error,
        )
        return

    try:
        completed = subprocess.run(
            command,
            cwd=str(repo_root),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        _append_check(
            report,
            name=check_name,
            status=FAIL,
            file=benchmark_file,
            message="Smoke test timed out after {} seconds.".format(timeout_seconds),
        )
        return

    _append_check(
        report,
        name=check_name,
        status=PASS if completed.returncode == 0 else FAIL,
        file=benchmark_file,
        message=(
            "Smoke test completed successfully using substituted LLM responses."
            if completed.returncode == 0 and mocked_llm
            else "Smoke test completed successfully."
            if completed.returncode == 0
            else "Smoke test failed with exit code {}.".format(completed.returncode)
        ),
        details=_summarize_output(completed.stdout),
    )


def _default_smoke_command(benchmark_file: str) -> List[str]:
    return [sys.executable, "benchmarking.py", "-f", benchmark_file]


def _configure_mock_runtime(
    env: Dict[str, str],
    repo_root: Path,
    example: Mapping[str, object],
) -> tuple:
    config = example.get("mock_runtime")
    if not isinstance(config, Mapping) or config.get("enabled") is not True:
        return False, ""

    python_paths = []
    for key in ("shared_pythonpath", "example_pythonpath"):
        values = config.get(key)
        if not isinstance(values, list) or not values:
            return True, "mock_runtime.{} must be a non-empty array.".format(key)
        for value in values:
            if not isinstance(value, str) or not value.strip():
                return True, "mock_runtime.{} must contain only strings.".format(key)
            path = (repo_root / _normalize_repo_path(value)).resolve()
            try:
                path.relative_to(repo_root.resolve())
            except ValueError:
                return True, "Mock Runtime path escapes the repository: {}".format(value)
            if not path.is_dir():
                return True, "Mock Runtime path does not exist: {}".format(value)
            python_paths.append(str(path))

    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        python_paths.append(existing_pythonpath)
    env["IANVS_LLM_MOCK"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    return True, ""


def _materialize_smoke_benchmark(
    repo_root: Path,
    benchmark_file: str,
    prepared_dataset_root: Optional[Path],
    temp_dir: Path,
) -> str:
    try:
        import yaml
    except ImportError:
        return benchmark_file

    benchmark_path = repo_root / benchmark_file
    benchmark = yaml.safe_load(benchmark_path.read_text(encoding="utf-8")) or {}
    job = benchmark.get("benchmarkingjob")
    if not isinstance(job, dict):
        return benchmark_file

    original_testenv = _normalize_repo_path(str(job.get("testenv", "")))
    if not original_testenv:
        return benchmark_file
    original_testenv_path = repo_root / original_testenv
    if not original_testenv_path.is_file():
        return benchmark_file

    testenv = yaml.safe_load(original_testenv_path.read_text(encoding="utf-8")) or {}
    if prepared_dataset_root:
        dataset_config = _dataset_config_from_testenv_payload(testenv)
        dataset = testenv.get("testenv", {}).get("dataset", {})
        if isinstance(dataset, dict):
            _rewrite_dataset_paths(dataset, dataset_config, prepared_dataset_root)

    smoke_testenv = temp_dir / "testenv.yaml"
    smoke_benchmark = temp_dir / "benchmarkingjob.yaml"
    smoke_workspace = temp_dir / "workspace"
    job["testenv"] = smoke_testenv.as_posix()
    job["workspace"] = smoke_workspace.as_posix()

    smoke_testenv.write_text(yaml.safe_dump(testenv, sort_keys=False), encoding="utf-8")
    smoke_benchmark.write_text(
        yaml.safe_dump(benchmark, sort_keys=False),
        encoding="utf-8",
    )
    return smoke_benchmark.as_posix()


def _dataset_config_from_testenv_payload(testenv: Mapping[str, object]) -> DatasetConfig:
    dataset = testenv.get("testenv", {}).get("dataset", {})
    if not isinstance(dataset, Mapping):
        return DatasetConfig(paths=[], root="")
    paths = [
        str(value)
        for value in dataset.values()
        if isinstance(value, str) and value.endswith(".jsonl")
    ]
    return DatasetConfig(paths=paths, root=_dataset_root_from_paths(paths))


def _rewrite_dataset_paths(
    dataset: Dict[str, object],
    dataset_config: DatasetConfig,
    prepared_dataset_root: Path,
) -> None:
    if not dataset_config.paths:
        return
    for key, value in list(dataset.items()):
        if not isinstance(value, str) or not value.endswith(".jsonl"):
            continue
        normalized = _normalize_repo_path(value)
        if dataset_config.root and _path_is_within(normalized, dataset_config.root):
            dataset[key] = (
                prepared_dataset_root
                / _relative_dataset_path(normalized, dataset_config.root)
            ).as_posix()
        else:
            dataset[key] = (prepared_dataset_root / Path(normalized).name).as_posix()


def _path_is_within(path: str, root: str) -> bool:
    normalized_path = _normalize_repo_path(path)
    normalized_root = _normalize_repo_path(root)
    return normalized_path == normalized_root or normalized_path.startswith(
        normalized_root.rstrip("/") + "/"
    )


def _relative_dataset_path(path: str, root: str) -> str:
    normalized_path = _normalize_repo_path(path)
    normalized_root = _normalize_repo_path(root)
    if not normalized_root:
        return normalized_path
    return normalized_path[len(normalized_root) :].lstrip("/")


def _looks_like_training_data_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return bool({"train", "training", "train_data", "training_data"} & parts)


def _safe_path_component(value: str) -> str:
    normalized = _normalize_repo_path(value)
    return "".join(char if char.isalnum() or char in "._-" else "-" for char in normalized)


def _summarize_output(output: str, max_lines: int = 60) -> List[str]:
    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    if len(lines) <= max_lines:
        return lines
    return lines[:max_lines] + ["... output truncated ..."]


def _optional_example_file(example: Mapping[str, object], key: str) -> Optional[str]:
    value = example.get(key)
    if not value:
        return None
    return _normalize_repo_path(str(value))


def _example_file(
    example: Mapping[str, object],
    key: str,
    example_path: str,
    default_name: str,
) -> str:
    value = _optional_example_file(example, key)
    if value:
        return value
    return _join_repo_path(example_path, default_name)


def _nested_file(example: Mapping[str, object], keys: Sequence[str]) -> Optional[str]:
    value: object = example
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    if not value:
        return None
    return _normalize_repo_path(str(value))


def _join_repo_path(*parts: str) -> str:
    return _normalize_repo_path("/".join(part.strip("/") for part in parts if part))


def _normalize_repo_path(value: str) -> str:
    value = value.strip().strip("\"'")
    if value.startswith("./"):
        value = value[2:]
    return value.rstrip("/")


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _append_issue_check(
    report: ExampleReport,
    name: str,
    issues: Sequence[str],
    fail_message: str,
    pass_message: str,
    file: str = "",
) -> None:
    _append_check(
        report,
        name=name,
        status=FAIL if issues else PASS,
        file=file,
        message=fail_message if issues else pass_message,
        details=issues,
    )


def _append_check(
    report: ExampleReport,
    name: str,
    status: str,
    message: str = "",
    file: str = "",
    details: Sequence[str] = (),
) -> None:
    report.checks.append(
        CheckResult(
            name=name,
            status=status,
            message=message,
            file=file,
            details=list(details),
        )
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ianvs smoke validation.")
    parser.add_argument(
        "--inventory",
        default=DEFAULT_INVENTORY_PATH,
        help="Example inventory YAML path.",
    )
    parser.add_argument(
        "--example",
        action="append",
        default=[],
        help="Example name or path to validate. Can be repeated.",
    )
    parser.add_argument("--all", action="store_true", help="Validate all examples.")
    parser.add_argument(
        "--jsonl-only",
        action="store_true",
        help="Only validate JSONL files from the existing configured dataset paths.",
    )
    parser.add_argument(
        "--no-execute",
        action="store_true",
        help="Prepare and validate smoke data without running Ianvs.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Smoke command timeout in seconds.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Report output format.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = Path.cwd()
    examples = load_inventory_examples(repo_root / args.inventory)
    selected_examples = _select_examples(examples, args.example, args.all)
    if args.jsonl_only:
        report = validate_jsonl_examples(
            repo_root=repo_root,
            examples=selected_examples,
            timeout_seconds=args.timeout,
        )
    else:
        report = validate_examples(
            repo_root=repo_root,
            examples=selected_examples,
            execute=not args.no_execute,
            timeout_seconds=args.timeout,
        )
    rendered = render_json(report) if args.format == "json" else render_smoke_markdown(report)
    sys.stdout.write(rendered)
    return 0 if report.passed else 1


def _select_examples(
    examples: Sequence[Mapping[str, object]],
    requested: Sequence[str],
    include_all: bool,
) -> List[Mapping[str, object]]:
    if include_all or not requested:
        return list(examples)
    selectors = {_normalize_repo_path(value) for value in requested}
    return [
        example
        for example in examples
        if _normalize_repo_path(str(example.get("name", ""))) in selectors
        or _normalize_repo_path(str(example.get("path", ""))) in selectors
        or _normalize_repo_path(str(example.get("benchmark_file", ""))) in selectors
    ]


if __name__ == "__main__":
    raise SystemExit(main())
