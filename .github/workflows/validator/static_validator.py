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

"""Static validation for Ianvs examples.

The checks in this module intentionally avoid executing examples. They inspect
inventory metadata, YAML configuration, and source text for common portability
problems described by the example restoration proposal.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Mapping, Optional, Sequence


PASS = "PASS"
FAIL = "FAIL"
ERROR = "ERROR"
WARNING = "WARNING"
SKIP = "SKIP"
YAML_SUFFIXES = (".yaml", ".yml")
BLOCKING_STATUSES = {ERROR, FAIL}
REPORT_DETAIL_STATUSES = BLOCKING_STATUSES | {WARNING, SKIP}

HARDCODED_PATH_RE = re.compile(
    r"(?:^|[\s'\"`=:(])(?P<path>(?:/(?!/)[^\s'\"`,)]+)|(?:[A-Za-z]:[\\/](?![\\/])[^\s'\"`,)]+))",
    re.IGNORECASE,
)
LOCAL_MODEL_RE = re.compile(
    r"(?i)(?:model(?:_path|_url)?|path)\s*[:=]\s*[\"'](?P<path>(?:/home/|/Users/|\.?/models?/)[^\"']*)"
)
CUDA_ONLY_RE = re.compile(
    r"(?i)(?:device\s*=\s*[\"']cuda[\"']|torch\.device\([\"']cuda[\"']\)|\.cuda\()"
)
REPO_PATH_RE = re.compile(
    r"(?P<path>(?:\./)?(?:examples|\.github|resources)/[A-Za-z0-9_./-]+"
    r"(?:\.yaml|\.yml|\.py|\.txt|\.json|\.jsonl|\.whl|\.zip)?)"
)
STATIC_VALIDATED_FILE_SUFFIXES = (".yaml", ".yml", ".py")


@dataclass
class CheckResult:
    name: str
    status: str
    message: str = ""
    file: str = ""
    details: List[str] = field(default_factory=list)


@dataclass
class ExampleReport:
    name: str
    path: str
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not any(check.status in BLOCKING_STATUSES for check in self.checks)


@dataclass
class StaticValidationReport:
    reports: List[ExampleReport]

    @property
    def passed(self) -> bool:
        return all(report.passed for report in self.reports)


def validate_examples(repo_root: Path, examples: Sequence[Mapping[str, object]]) -> StaticValidationReport:
    reports = [validate_example(repo_root, example) for example in examples]
    return StaticValidationReport(reports=reports)


def validate_example(repo_root: Path, example: Mapping[str, object]) -> ExampleReport:
    example_path = _normalize_repo_path(str(example.get("path", "")))
    example_name = str(example.get("name") or example_path)
    report = ExampleReport(name=example_name, path=example_path)
    root = repo_root / example_path

    _check_path_exists(
        report,
        "Example directory exists",
        repo_root,
        example_path,
        is_dir=True,
    )
    benchmark_file = _example_file(example, "benchmark_file", example_path, "benchmarkingjob.yaml")
    requirements_file = _optional_example_file(example, "requirements_file")
    prepare_script = _nested_file(example, ("dataset", "prepare_script"))

    _check_path_exists(report, "benchmarkingjob.yaml exists", repo_root, benchmark_file)
    if requirements_file:
        _check_path_exists(report, "requirements file exists", repo_root, requirements_file)
    if prepare_script:
        _check_path_exists(report, "dataset prepare script exists", repo_root, prepare_script)
    _check_prepare_env_contract(report, repo_root, example)
    _check_mock_runtime_contract(report, repo_root, example)

    files = _example_files(root)
    _check_yaml_syntax(report, files)
    _check_repo_path_references(report, repo_root, files)
    _check_hardcoded_paths(report, repo_root, files)
    _check_local_model_paths(report, repo_root, files)
    _check_cuda_only_assumptions(report, repo_root, files)
    _check_metric_empty_pair_guard(report, repo_root, example_path)

    return report


def _check_prepare_env_contract(
    report: ExampleReport,
    repo_root: Path,
    example: Mapping[str, object],
) -> None:
    config = example.get("prepare_env")
    if config is None:
        return
    issues = []
    if not isinstance(config, Mapping):
        issues.append("prepare_env must be an object")
    else:
        working_directory = config.get("working_directory")
        steps = config.get("steps")
        if not isinstance(working_directory, str) or not working_directory.strip():
            issues.append("prepare_env.working_directory must be a non-empty string")
        elif not (repo_root / _normalize_repo_path(working_directory)).is_dir():
            issues.append("working directory is missing: {}".format(working_directory))
        if not isinstance(steps, list) or not steps:
            issues.append("prepare_env.steps must be a non-empty array")
        else:
            for index, step in enumerate(steps):
                prefix = "prepare_env.steps[{}]".format(index)
                if not isinstance(step, Mapping):
                    issues.append("{} must be an object".format(prefix))
                    continue
                missing = [
                    key
                    for key in ("name", "type", "script", "args", "timeout")
                    if key not in step
                ]
                if missing:
                    issues.append("{} missing: {}".format(prefix, ", ".join(missing)))
                    continue
                args = step.get("args")
                if not isinstance(args, list) or any(
                    not isinstance(arg, str) for arg in args
                ):
                    issues.append("{}.args must be an array of strings".format(prefix))
                timeout = step.get("timeout")
                if (
                    isinstance(timeout, bool)
                    or not isinstance(timeout, int)
                    or timeout <= 0
                ):
                    issues.append("{}.timeout must be a positive integer".format(prefix))
                script = step.get("script")
                if (
                    isinstance(working_directory, str)
                    and isinstance(script, str)
                    and not (
                        repo_root
                        / _normalize_repo_path(working_directory)
                        / _normalize_repo_path(script)
                    ).is_file()
                ):
                    issues.append("{} script is missing: {}".format(prefix, script))
    _append_issue_check(
        report,
        name="Environment preparation contract",
        issues=issues,
        fail_message=(
            "The prepare_env contract is invalid. Impact: automated environment "
            "setup may fail before validation starts."
        ),
        pass_message="The prepare_env contract and scripts are valid.",
    )


def _check_mock_runtime_contract(
    report: ExampleReport,
    repo_root: Path,
    example: Mapping[str, object],
) -> None:
    config = example.get("mock_runtime")
    if config is None:
        return
    issues = []
    if not isinstance(config, Mapping):
        issues.append("mock_runtime must be an object")
    elif config.get("enabled") is True:
        for key in ("shared_pythonpath", "example_pythonpath"):
            values = config.get(key)
            if not isinstance(values, list) or not values:
                issues.append("mock_runtime.{} must be a non-empty array".format(key))
                continue
            for value in values:
                if not isinstance(value, str) or not value.strip():
                    issues.append("mock_runtime.{} contains an invalid path".format(key))
                elif not (repo_root / _normalize_repo_path(value)).is_dir():
                    issues.append("mock runtime path is missing: {}".format(value))
    _append_issue_check(
        report,
        name="Mock LLM runtime contract",
        issues=issues,
        fail_message=(
            "The Mock LLM runtime metadata is invalid. Impact: offline runtime "
            "smoke validation may fail to start."
        ),
        pass_message="The Mock LLM runtime paths are valid.",
    )


def render_markdown(report: StaticValidationReport) -> str:
    lines = ["# Static Validation Report", ""]
    lines.append("Overall result: {}".format(_report_result(report)))

    for example_report in report.reports:
        lines.extend(["", "## Example", "", example_report.path, "", "### Validation Result", ""])
        lines.extend(["| Check | Result |", "|---|---|"])
        for check in example_report.checks:
            lines.append("| {} | {} |".format(_escape_table(check.name), check.status))

        failures = [
            check
            for check in example_report.checks
            if check.status in REPORT_DETAIL_STATUSES
        ]
        if failures:
            lines.extend(["", "### Details", ""])
            for check in failures:
                lines.append("#### {}".format(check.name))
                lines.append("")
                lines.append("Result: {}".format(check.status))
                if check.file:
                    lines.append("")
                    lines.append("File: `{}`".format(check.file))
                if check.message:
                    lines.append("")
                    lines.append(check.message)
                if check.details:
                    lines.append("")
                    lines.append("Detected:")
                    for detail in check.details:
                        lines.append("- `{}`".format(detail))
                lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _report_result(report: StaticValidationReport) -> str:
    if any(
        check.status == ERROR
        for example_report in report.reports
        for check in example_report.checks
    ):
        return ERROR
    if any(
        check.status == WARNING
        for example_report in report.reports
        for check in example_report.checks
    ):
        return WARNING
    return PASS if report.passed else FAIL


def render_json(report: StaticValidationReport) -> str:
    def example_executed(example: ExampleReport) -> bool:
        return any(check.status != SKIP for check in example.checks)

    def example_passed(example: ExampleReport) -> bool:
        return example.passed and example_executed(example)

    payload = {
        "passed": report.passed and all(
            example_passed(example) for example in report.reports
        ),
        "examples": [
            {
                "name": example.name,
                "path": example.path,
                "passed": example_passed(example),
                "executed": example_executed(example),
                "checks": [
                    {
                        "name": check.name,
                        "status": check.status,
                        "message": check.message,
                        "file": check.file,
                        "details": check.details,
                    }
                    for check in example.checks
                ],
            }
            for example in report.reports
        ],
    }
    return json.dumps(payload, indent=2)


def _check_path_exists(
    report: ExampleReport,
    name: str,
    repo_root: Path,
    repo_path: str,
    is_dir: bool = False,
) -> None:
    path = repo_root / repo_path
    exists = path.is_dir() if is_dir else path.is_file()
    details = ()
    if not exists:
        # A missing path has no source line to point at.  Make that explicit so
        # consumers do not mistake the path itself for a line-level finding.
        details = ("{} -> (Line N/A): required path is missing".format(repo_path),)
    _append_check(
        report,
        name=name,
        status=PASS if exists else ERROR,
        file=repo_path,
        message=(
            "Required path exists."
            if exists
            else (
                "Required path is missing. Impact: the example cannot be validated "
                "or executed from a clean checkout."
            )
        ),
        details=details,
    )


def _check_yaml_syntax(report: ExampleReport, files: Sequence[Path]) -> None:
    yaml_files = [path for path in files if path.suffix in YAML_SUFFIXES]
    try:
        import yaml
    except ImportError:
        _append_check(
            report,
            name="YAML syntax",
            status=SKIP,
            message="PyYAML is unavailable; YAML syntax validation was skipped.",
        )
        return

    failures = []
    for path in yaml_files:
        try:
            yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            failures.append(_format_yaml_error(path, exc))

    _append_issue_check(
        report,
        name="YAML syntax",
        issues=failures,
        fail_message=(
            "Invalid YAML syntax found. Impact: Ianvs cannot parse the affected "
            "configuration."
        ),
        pass_message="All YAML files parsed successfully.",
        failure_status=ERROR,
    )


def _check_repo_path_references(
    report: ExampleReport,
    repo_root: Path,
    files: Sequence[Path],
) -> None:
    missing_code_or_config = []
    missing_parent_references = []
    for path in files:
        text = _read_text(path)
        for match in REPO_PATH_RE.finditer(text):
            repo_path = _normalize_repo_path(match.group("path"))
            if _looks_like_generated_dataset_path(repo_path):
                continue
            if (repo_root / repo_path).exists():
                continue

            detail = _format_detected_value(
                path,
                _line_number_for_match(text, match, "path"),
                repo_path,
            )
            if _is_code_or_config_reference(repo_path):
                missing_code_or_config.append(detail)
                continue

            parent_path = _referenced_parent_path(repo_path)
            if parent_path and not (repo_root / parent_path).is_dir():
                missing_parent_references.append(
                    "{} (parent folder missing: {})".format(detail, parent_path)
                )

    _append_issue_check(
        report,
        name="Repository path references exist",
        issues=sorted(set(missing_code_or_config)),
        fail_message=(
            "Broken repository-local Python/YAML path references found. Impact: "
            "Python modules or YAML configurations may fail to load."
        ),
        pass_message="Repository-local Python/YAML path references resolve.",
        failure_status=ERROR,
    )
    _append_issue_check(
        report,
        name="Repository path parent references exist",
        issues=sorted(set(missing_parent_references)),
        fail_message=(
            "Repository-local non-code path references have missing parent folders. "
            "Impact: datasets or resources may not be created or resolved correctly."
        ),
        pass_message="Repository-local non-code path reference parent folders resolve.",
        failure_status=WARNING,
    )


def _check_hardcoded_paths(
    report: ExampleReport,
    repo_root: Path,
    files: Sequence[Path],
) -> None:
    matches = _collect_regex_matches(repo_root, files, HARDCODED_PATH_RE)
    _append_issue_check(
        report,
        name="Hardcoded local path check",
        issues=matches,
        fail_message=(
            "Contributor-specific absolute paths were found. Impact: the example "
            "may work only on the contributor's machine."
        ),
        pass_message="No contributor-specific absolute paths found.",
        failure_status=WARNING,
    )


def _check_local_model_paths(
    report: ExampleReport,
    repo_root: Path,
    files: Sequence[Path],
) -> None:
    matches = _collect_regex_matches(repo_root, files, LOCAL_MODEL_RE)
    _append_issue_check(
        report,
        name="Local model path check",
        issues=matches,
        fail_message=(
            "Local-only model path references were found. Impact: clean CI runners "
            "and other developers may not be able to load the model."
        ),
        pass_message="No local-only model paths found.",
        failure_status=WARNING,
    )


def _check_cuda_only_assumptions(
    report: ExampleReport,
    repo_root: Path,
    files: Sequence[Path],
) -> None:
    failures = []
    for path in files:
        if path.suffix != ".py":
            continue
        text = _read_text(path)
        match = CUDA_ONLY_RE.search(text)
        if not match:
            continue
        has_fallback = "torch.cuda.is_available()" in text and "cpu" in text
        if has_fallback:
            continue
        failures.append(
            _format_detected_value(
                path,
                _line_number_for_match(text, match),
                match.group(0),
            )
        )

    _append_issue_check(
        report,
        name="CUDA-only device check",
        issues=failures,
        fail_message=(
            "CUDA-only device assumptions were found. Impact: the example may fail "
            "on runners without a CUDA-capable GPU."
        ),
        pass_message="No CUDA-only device assumptions found.",
        failure_status=WARNING,
    )


def _check_metric_empty_pair_guard(report: ExampleReport, repo_root: Path, example_path: str) -> None:
    metric_dir = repo_root / example_path / "testenv"
    metric_files = list(metric_dir.glob("*.py")) if metric_dir.is_dir() else []
    if not metric_files:
        _append_check(
            report,
            name="Metric empty-pair guard",
            status=SKIP,
            message="No metric Python files were found.",
        )
        return

    risky = []
    guarded = []
    for path in metric_files:
        text = _read_text(path)
        if "/ len(" not in text:
            continue
        has_guard = "if same_elements else 0.0" in text or "if len(" in text
        if has_guard:
            guarded.append(_repo_display_path(path))
        else:
            for match in re.finditer(r"/\s*len\(", text):
                risky.append(
                    _format_detected_value(
                        path,
                        _line_number_for_match(text, match),
                        match.group(0),
                    )
                )

    _append_issue_check(
        report,
        name="Metric empty-pair guard",
        issues=risky,
        fail_message=(
            "Metric division may crash on empty prediction-answer pairs. Impact: "
            "evaluation may raise a division-by-zero error when no pairs are available."
        ),
        pass_message="Metric files include an empty-pair guard or do not divide by a collection length.",
        pass_details=guarded,
        failure_status=WARNING,
    )


def _collect_regex_matches(
    repo_root: Path,
    files: Sequence[Path],
    pattern: re.Pattern,
) -> List[str]:
    matches = []
    for path in files:
        text = _read_text(path)
        for match in pattern.finditer(text):
            value = match.groupdict().get("path") or match.group(0)
            matches.append(
                _format_detected_value(
                    path,
                    _line_number_for_match(text, match, "path"),
                    value,
                )
            )
    return sorted(set(matches))


def _format_detected_value(path: Path, line_number: int, value: str) -> str:
    return "{} -> (Line {}): {}".format(_repo_display_path(path), line_number, value)


def _format_yaml_error(path: Path, error: object) -> str:
    """Render a YAML parser error as a single line-level diagnostic."""
    mark = getattr(error, "problem_mark", None)
    if mark is not None and hasattr(mark, "line"):
        return "{} -> (Line {}): {}".format(
            _repo_display_path(path),
            int(mark.line) + 1,
            getattr(error, "problem", str(error)),
        )
    return "{} -> (Line N/A): {}".format(_repo_display_path(path), error)


def _line_number_for_match(text: str, match: re.Match, group_name: str = "") -> int:
    start = -1
    if group_name:
        try:
            start = match.start(group_name)
        except IndexError:
            start = -1
    if start < 0:
        start = match.start()
    return text.count("\n", 0, start) + 1


def _append_issue_check(
    report: ExampleReport,
    name: str,
    issues: Sequence[str],
    fail_message: str,
    pass_message: str,
    pass_details: Sequence[str] = (),
    failure_status: str = ERROR,
) -> None:
    has_issues = bool(issues)
    _append_check(
        report,
        name=name,
        status=failure_status if has_issues else PASS,
        message=fail_message if has_issues else pass_message,
        details=issues if has_issues else pass_details,
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


def _example_files(root: Path) -> List[Path]:
    if not root.is_dir():
        return []
    ignored_parts = {"__pycache__", ".pytest_cache"}
    files = []
    for path in root.rglob("*"):
        if not path.is_file() or ignored_parts.intersection(path.parts):
            continue
        if path.suffix.lower() not in STATIC_VALIDATED_FILE_SUFFIXES:
            continue
        files.append(path)
    return files


def _optional_example_file(
    example: Mapping[str, object],
    key: str,
) -> Optional[str]:
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


def _looks_like_generated_dataset_path(repo_path: str) -> bool:
    return "/dataset/" in repo_path


def _is_code_or_config_reference(repo_path: str) -> bool:
    return Path(repo_path).suffix.lower() in (".py", ".yaml", ".yml")


def _referenced_parent_path(repo_path: str) -> str:
    parent = Path(repo_path).parent.as_posix()
    return "" if parent == "." else _normalize_repo_path(parent)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def _repo_display_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|")
