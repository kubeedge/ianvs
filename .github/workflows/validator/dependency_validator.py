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

"""Dependency validation for Ianvs examples.

The default checks are offline and deterministic: dependency file presence,
requirements syntax, Python marker coverage for the supported CI matrix, and
obvious undeclared third-party imports. A pip install check is available as an
explicit opt-in because it may require network access and can be expensive for
ML examples.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

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


DEFAULT_PYTHON_MATRIX = ("3.8", "3.9", "3.10")
INSTALL_MODE_SKIP = "skip"
INSTALL_MODE_DRY_RUN = "dry-run"
INSTALL_MODE_INSTALL = "install"
INSTALL_MODES = (INSTALL_MODE_SKIP, INSTALL_MODE_DRY_RUN, INSTALL_MODE_INSTALL)
PYTHON_FILE_SUFFIX = ".py"
REQUIREMENTS_OPTIONS = ("-", "--")
REQUIREMENTS_REFERENCE_PREFIXES = ("-r", "--requirement", "-c", "--constraint")
PROJECT_PROVIDED_IMPORTS = {
    "core",
    "examples",
    "sedna",
    "testalgorithms",
    "testenv",
}
IMPORT_PACKAGE_ALIASES = {
    "PIL": "pillow",
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
}


def validate_examples(
    repo_root: Path,
    examples: Sequence[Mapping[str, object]],
    python_versions: Sequence[str] = DEFAULT_PYTHON_MATRIX,
    install_mode: str = INSTALL_MODE_SKIP,
    timeout_seconds: int = 300,
) -> StaticValidationReport:
    reports = [
        validate_example(
            repo_root=repo_root,
            example=example,
            python_versions=python_versions,
            install_mode=install_mode,
            timeout_seconds=timeout_seconds,
        )
        for example in examples
    ]
    return StaticValidationReport(reports=reports)


def validate_example(
    repo_root: Path,
    example: Mapping[str, object],
    python_versions: Sequence[str],
    install_mode: str,
    timeout_seconds: int,
) -> ExampleReport:
    example_path = _normalize_repo_path(str(example.get("path", "")))
    example_name = str(example.get("name") or example_path)
    report = ExampleReport(name=example_name, path=example_path)

    requirements_file = _optional_example_file(example, "requirements_file")
    if not requirements_file:
        _append_check(
            report,
            name="Dependency file declared",
            status=SKIP,
            message="No example-specific dependency file is declared in the inventory.",
        )
        return report

    requirements_path = repo_root / requirements_file
    _append_check(
        report,
        name="Dependency file exists",
        status=PASS if requirements_path.is_file() else FAIL,
        file=requirements_file,
        message=(
            "Dependency file exists."
            if requirements_path.is_file()
            else "Declared dependency file is missing."
        ),
    )
    if not requirements_path.is_file():
        return report

    requirement_lines = _read_requirement_lines(requirements_path)
    _check_dependency_file_not_empty(report, requirement_lines, requirements_file)
    parsed_requirements, syntax_errors = _parse_requirements(
        requirement_lines,
        requirements_path.parent,
    )
    _append_issue_check(
        report,
        name="Dependency declaration syntax",
        issues=syntax_errors,
        fail_message="Invalid requirement declarations were found.",
        pass_message="Dependency declarations parsed successfully.",
        file=requirements_file,
    )
    _check_python_marker_coverage(
        report,
        parsed_requirements,
        python_versions,
        requirements_file,
    )
    _check_declared_runtime_imports(
        report,
        repo_root,
        example,
        parsed_requirements,
    )
    _check_pip_install(
        report,
        repo_root,
        requirements_file,
        install_mode,
        timeout_seconds,
    )
    return report


def render_dependency_markdown(report: StaticValidationReport) -> str:
    rendered = render_markdown(report)
    return rendered.replace("# Static Validation Report", "# Dependency Validation Report", 1)


def _check_dependency_file_not_empty(
    report: ExampleReport,
    requirement_lines: Sequence[Tuple[int, str]],
    requirements_file: str,
) -> None:
    _append_check(
        report,
        name="Dependency file is not empty",
        status=PASS if requirement_lines else FAIL,
        file=requirements_file,
        message=(
            "Dependency file contains installable declarations."
            if requirement_lines
            else "Dependency file does not contain any installable declarations."
        ),
    )


def _parse_requirements(
    requirement_lines: Sequence[Tuple[int, str]],
    base_dir: Path,
) -> Tuple[List[object], List[str]]:
    requirement_cls = _requirement_parser()
    requirements = []
    errors = []

    for line_number, raw_value in requirement_lines:
        value = _strip_inline_comment(raw_value).strip()
        if not value:
            continue
        if _is_requirements_reference(value):
            referenced_path = _referenced_requirements_path(value, base_dir)
            if referenced_path and not referenced_path.is_file():
                errors.append(
                    "line {}: referenced dependency file is missing: {}".format(
                        line_number,
                        referenced_path.as_posix(),
                    )
                )
            continue
        if value.startswith(REQUIREMENTS_OPTIONS):
            continue
        if _looks_like_direct_reference(value):
            continue

        try:
            requirements.append(requirement_cls(value))
        except Exception as exc:  # pragma: no cover - parser error type varies.
            errors.append("line {}: {} ({})".format(line_number, value, exc))

    return requirements, errors


def _check_python_marker_coverage(
    report: ExampleReport,
    requirements: Sequence[object],
    python_versions: Sequence[str],
    requirements_file: str,
) -> None:
    excluded = []
    for requirement in requirements:
        marker = getattr(requirement, "marker", None)
        if marker is None:
            continue
        supported_versions = [
            version
            for version in python_versions
            if marker.evaluate({"python_version": version})
        ]
        if not supported_versions:
            excluded.append(str(requirement))

    _append_issue_check(
        report,
        name="Python version marker compatibility",
        issues=excluded,
        fail_message="Some requirements are excluded for every supported Python version.",
        pass_message="Requirement markers support at least one target Python version.",
        file=requirements_file,
    )


def _check_declared_runtime_imports(
    report: ExampleReport,
    repo_root: Path,
    example: Mapping[str, object],
    requirements: Sequence[object],
) -> None:
    imported_modules = _collect_external_imports(repo_root, example)
    declared_packages = _declared_packages(requirements)
    project_packages = _declared_project_packages(repo_root)

    missing = []
    for module_name in sorted(imported_modules):
        package_name = _normalize_package_name(
            IMPORT_PACKAGE_ALIASES.get(module_name, module_name)
        )
        if package_name in declared_packages or package_name in project_packages:
            continue
        missing.append(module_name)

    _append_issue_check(
        report,
        name="Runtime imports declared",
        issues=missing,
        fail_message="Third-party imports are not declared by the example or project requirements.",
        pass_message="Example runtime imports are covered by dependency declarations.",
    )


def _check_pip_install(
    report: ExampleReport,
    repo_root: Path,
    requirements_file: str,
    install_mode: str,
    timeout_seconds: int,
) -> None:
    if install_mode == INSTALL_MODE_SKIP:
        _append_check(
            report,
            name="pip install check",
            status=SKIP,
            file=requirements_file,
            message="pip install validation was not requested.",
        )
        return

    pip_bootstrap = _ensure_pip_available(repo_root, timeout_seconds)
    if pip_bootstrap:
        _append_check(
            report,
            name="pip availability",
            status=PASS if pip_bootstrap["returncode"] == 0 else FAIL,
            message=pip_bootstrap["message"],
            details=pip_bootstrap["details"],
        )
        if pip_bootstrap["returncode"] != 0:
            return

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "-r",
        requirements_file,
    ]
    if install_mode == INSTALL_MODE_DRY_RUN:
        command.insert(4, "--dry-run")

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
            name="pip install check",
            status=FAIL,
            file=requirements_file,
            message="pip install validation timed out after {} seconds.".format(
                timeout_seconds
            ),
        )
        return

    output = _summarize_output(completed.stdout)
    _append_check(
        report,
        name="pip install check",
        status=PASS if completed.returncode == 0 else FAIL,
        file=requirements_file,
        message=(
            "pip {} completed successfully.".format(install_mode)
            if completed.returncode == 0
            else "pip {} failed with exit code {}.".format(
                install_mode,
                completed.returncode,
            )
        ),
        details=output,
    )


def _ensure_pip_available(repo_root: Path, timeout_seconds: int) -> Optional[Dict[str, object]]:
    check_command = [sys.executable, "-m", "pip", "--version"]
    completed = subprocess.run(
        check_command,
        cwd=str(repo_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode == 0:
        return None

    bootstrap_command = [sys.executable, "-m", "ensurepip", "--upgrade"]
    try:
        bootstrap = subprocess.run(
            bootstrap_command,
            cwd=str(repo_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "returncode": 1,
            "message": "pip is unavailable and ensurepip timed out after {} seconds.".format(
                timeout_seconds
            ),
            "details": _summarize_output(completed.stdout),
        }

    return {
        "returncode": bootstrap.returncode,
        "message": (
            "pip was bootstrapped with ensurepip."
            if bootstrap.returncode == 0
            else "pip is unavailable and ensurepip failed with exit code {}.".format(
                bootstrap.returncode
            )
        ),
        "details": _summarize_output(completed.stdout + "\n" + bootstrap.stdout),
    }


def _collect_external_imports(
    repo_root: Path,
    example: Mapping[str, object],
) -> Set[str]:
    example_root = repo_root / _normalize_repo_path(str(example.get("path", "")))
    if not example_root.is_dir():
        return set()

    imports = set()
    local_modules = _local_module_names(example_root)
    python_files = _runtime_python_files(repo_root, example)
    if not python_files:
        python_files = [
            path
            for path in example_root.rglob("*{}".format(PYTHON_FILE_SUFFIX))
            if "__pycache__" not in path.parts
        ]

    for path in python_files:
        if "__pycache__" in path.parts:
            continue
        for module_name in _imports_from_python_file(path):
            if module_name in local_modules:
                continue
            if module_name in PROJECT_PROVIDED_IMPORTS:
                continue
            if _is_stdlib_module(module_name):
                continue
            imports.add(module_name)
    return imports


def _runtime_python_files(
    repo_root: Path,
    example: Mapping[str, object],
) -> List[Path]:
    try:
        import yaml
    except ImportError:
        return []

    benchmark_file = _optional_example_file(example, "benchmark_file")
    if not benchmark_file:
        return []
    benchmark_path = repo_root / benchmark_file
    if not benchmark_path.is_file():
        return []

    benchmark = yaml.safe_load(benchmark_path.read_text(encoding="utf-8")) or {}
    job = benchmark.get("benchmarkingjob", {})
    files = []

    testenv_path = _normalize_repo_path(str(job.get("testenv", "")))
    if testenv_path:
        files.extend(_testenv_python_files(repo_root, testenv_path, yaml))

    test_object = job.get("test_object", {})
    for algorithm in test_object.get("algorithms", []) or []:
        algorithm_path = _normalize_repo_path(str(algorithm.get("url", "")))
        files.extend(_algorithm_python_files(repo_root, algorithm_path, yaml))

    return _unique_existing_files(files)


def _testenv_python_files(repo_root: Path, testenv_path: str, yaml_module) -> List[Path]:
    path = repo_root / testenv_path
    if not path.is_file():
        return []
    payload = yaml_module.safe_load(path.read_text(encoding="utf-8")) or {}
    metrics = payload.get("testenv", {}).get("metrics", []) or []
    return [
        repo_root / _normalize_repo_path(str(metric.get("url", "")))
        for metric in metrics
        if str(metric.get("url", "")).endswith(PYTHON_FILE_SUFFIX)
    ]


def _algorithm_python_files(repo_root: Path, algorithm_path: str, yaml_module) -> List[Path]:
    path = repo_root / algorithm_path
    if not path.is_file():
        return []
    payload = yaml_module.safe_load(path.read_text(encoding="utf-8")) or {}
    modules = payload.get("algorithm", {}).get("modules", []) or []
    return [
        repo_root / _normalize_repo_path(str(module.get("url", "")))
        for module in modules
        if str(module.get("url", "")).endswith(PYTHON_FILE_SUFFIX)
    ]


def _unique_existing_files(paths: Sequence[Path]) -> List[Path]:
    seen = set()
    unique = []
    for path in paths:
        if not path.is_file():
            continue
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _imports_from_python_file(path: Path) -> Set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return set()

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imports.add(node.module.split(".", 1)[0])
    return imports


def _local_module_names(example_root: Path) -> Set[str]:
    modules = set()
    for path in example_root.rglob("*.py"):
        modules.add(path.stem)
        if path.name == "__init__.py":
            modules.add(path.parent.name)
    return modules


def _declared_packages(requirements: Sequence[object]) -> Set[str]:
    return {
        _normalize_package_name(str(getattr(requirement, "name", "")))
        for requirement in requirements
        if getattr(requirement, "name", "")
    }


def _declared_project_packages(repo_root: Path) -> Set[str]:
    requirements_path = repo_root / "requirements.txt"
    if not requirements_path.is_file():
        return set()
    requirements, _ = _parse_requirements(
        _read_requirement_lines(requirements_path),
        requirements_path.parent,
    )
    return _declared_packages(requirements)


def _read_requirement_lines(path: Path) -> List[Tuple[int, str]]:
    lines = []
    for index, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        lines.append((index, line))
    return lines


def _requirement_parser():
    try:
        from packaging.requirements import Requirement
    except ImportError:
        from pip._vendor.packaging.requirements import Requirement
    return Requirement


def _strip_inline_comment(value: str) -> str:
    return re.sub(r"\s+#.*$", "", value)


def _is_requirements_reference(value: str) -> bool:
    return value.startswith(REQUIREMENTS_REFERENCE_PREFIXES)


def _referenced_requirements_path(value: str, base_dir: Path) -> Optional[Path]:
    parts = value.split()
    if len(parts) < 2:
        return None
    return base_dir / parts[1]


def _looks_like_direct_reference(value: str) -> bool:
    return "://" in value or value.endswith((".whl", ".zip", ".tar.gz"))


def _normalize_package_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _is_stdlib_module(module_name: str) -> bool:
    stdlib_names = getattr(sys, "stdlib_module_names", None)
    if stdlib_names is not None:
        return module_name in stdlib_names
    return module_name in {
        "argparse",
        "__future__",
        "collections",
        "contextlib",
        "csv",
        "datetime",
        "functools",
        "glob",
        "importlib",
        "itertools",
        "json",
        "logging",
        "math",
        "os",
        "pathlib",
        "re",
        "shutil",
        "subprocess",
        "sys",
        "tempfile",
        "time",
        "typing",
        "unittest",
        "zipfile",
    }


def _summarize_output(output: str, max_lines: int = 40) -> List[str]:
    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    if len(lines) <= max_lines:
        return lines
    return lines[:max_lines] + ["... output truncated ..."]


def _optional_example_file(example: Mapping[str, object], key: str) -> Optional[str]:
    value = example.get(key)
    if not value:
        return None
    return _normalize_repo_path(str(value))


def _normalize_repo_path(value: str) -> str:
    value = value.strip().strip("\"'")
    if value.startswith("./"):
        value = value[2:]
    return value.rstrip("/")


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
    parser = argparse.ArgumentParser(description="Run Ianvs dependency validation.")
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
        "--python-version",
        action="append",
        default=[],
        help="Supported Python version. Can be repeated.",
    )
    parser.add_argument(
        "--install-mode",
        choices=INSTALL_MODES,
        default=INSTALL_MODE_SKIP,
        help="Whether to skip, dry-run, or execute pip install validation.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="pip install validation timeout in seconds.",
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
    report = validate_examples(
        repo_root=repo_root,
        examples=selected_examples,
        python_versions=args.python_version or DEFAULT_PYTHON_MATRIX,
        install_mode=args.install_mode,
        timeout_seconds=args.timeout,
    )
    rendered = render_json(report) if args.format == "json" else render_dependency_markdown(report)
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
