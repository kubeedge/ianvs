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

"""Entry point for local and CI example validation."""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

from dependency_validator import (
    INSTALL_MODE_DRY_RUN,
    INSTALL_MODE_INSTALL,
    INSTALL_MODE_SKIP,
)
from dependency_validator import validate_examples as validate_dependencies
from smoke_test_validator import (
    DEFAULT_TIMEOUT_SECONDS,
    prepare_example_environments,
    validate_examples as validate_smoke_examples,
)
from smoke_test_validator import validate_jsonl_examples
from services.inventory_loader import DEFAULT_INVENTORY_PATH, load_inventory_examples
from static_validator import (
    ERROR,
    SKIP,
    CheckResult,
    ExampleReport,
    StaticValidationReport,
    render_json,
    render_markdown,
)
from static_validator import validate_examples as validate_static_examples

REPORT_FORMAT_JSON = "json"
REPORT_FORMAT_MARKDOWN = "markdown"
LOGGER = logging.getLogger("ianvs.validator.runner")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Ianvs example validation modules."
    )
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
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate all active examples in the inventory.",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="Run Tier 0 static validation.",
    )
    parser.add_argument(
        "--dependency",
        action="store_true",
        help="Run dependency declaration validation.",
    )
    parser.add_argument(
        "--prepare-env",
        action="store_true",
        help="Run ordered inventory-defined environment preparation steps.",
    )
    parser.add_argument(
        "--format",
        choices=(REPORT_FORMAT_MARKDOWN, REPORT_FORMAT_JSON),
        default=REPORT_FORMAT_MARKDOWN,
        help="Report output format.",
    )
    parser.add_argument(
        "--report",
        default="",
        help="Optional path to write the validation report.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run runtime smoke validation.",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Run JSONL dataset validation.",
    )
    parser.add_argument(
        "--pip-install-check",
        action="store_true",
        help="Run pip install --dry-run for example dependency files.",
    )
    parser.add_argument(
        "--pip-install",
        action="store_true",
        help="Run real pip install for example dependency files.",
    )
    parser.add_argument(
        "--no-execute-smoke",
        action="store_true",
        help="Prepare and validate smoke data without running Ianvs.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Timeout in seconds for dependency install checks and smoke commands.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = Path.cwd()
    selected_examples = load_selected_examples(repo_root, args)

    if not selected_examples:
        LOGGER.error("No inventory examples matched the requested selection.")
        return 1

    try:
        report = run_validation_pipeline(repo_root, selected_examples, args)
    except Exception as error:  # pragma: no cover - traceback is diagnostic output
        traceback.print_exc()
        report = unexpected_failure_report(selected_examples, error)
    rendered = render_report(report, args.format)
    write_or_print_report(rendered, args.report)
    return 0 if report.passed else 1


def unexpected_failure_report(
    examples: Sequence[Mapping[str, object]],
    error: Exception,
) -> StaticValidationReport:
    """Convert an unexpected validator crash into a blocking result artifact."""
    detail = "{}: {}".format(type(error).__name__, error)
    reports = []
    for example in examples:
        example_path = normalize_selector(str(example.get("path", "")))
        example_name = str(example.get("name") or example_path)
        reports.append(
            ExampleReport(
                name=example_name,
                path=example_path,
                checks=[
                    CheckResult(
                        name="Validation runner internal error",
                        status=ERROR,
                        message=(
                            "Validation stopped because an unexpected exception "
                            "was raised."
                        ),
                        file=str(example.get("benchmark_file") or example_path),
                        details=[detail],
                    )
                ],
            )
        )
    return StaticValidationReport(reports=reports)


def load_selected_examples(
    repo_root: Path,
    args: argparse.Namespace,
) -> List[Mapping[str, object]]:
    inventory_path = repo_root / args.inventory
    examples = load_inventory_examples(
        inventory_path,
        active_only=not bool(args.example),
    )
    return select_examples(examples, args.example, args.all)


def run_validation_pipeline(
    repo_root: Path,
    selected_examples: Sequence[Mapping[str, object]],
    args: argparse.Namespace,
) -> StaticValidationReport:
    reports: List[StaticValidationReport] = []
    dynamic_examples = selected_examples
    if not runs_static_validation(args):
        dynamic_examples = active_examples(selected_examples)
        skipped_examples = inactive_examples(selected_examples)
        if skipped_examples:
            reports.append(skip_dynamic_examples(skipped_examples))

    if runs_static_validation(args):
        reports.append(
            validate_static_examples(repo_root=repo_root, examples=selected_examples)
        )
    if args.dependency and dynamic_examples:
        reports.append(
            validate_dependencies(
                repo_root=repo_root,
                examples=dynamic_examples,
                install_mode=dependency_install_mode(args),
                timeout_seconds=args.timeout,
            )
        )
    if args.prepare_env and dynamic_examples:
        reports.append(
            prepare_example_environments(
                repo_root=repo_root,
                examples=dynamic_examples,
            )
        )
    if args.jsonl and dynamic_examples:
        reports.append(
            validate_jsonl_examples(
                repo_root=repo_root,
                examples=dynamic_examples,
                timeout_seconds=args.timeout,
            )
        )
    if args.smoke and dynamic_examples:
        reports.append(
            validate_smoke_examples(
                repo_root=repo_root,
                examples=dynamic_examples,
                execute=not args.no_execute_smoke,
                timeout_seconds=args.timeout,
            )
        )

    return merge_reports(reports)


def runs_dynamic_validation(args: argparse.Namespace) -> bool:
    return any((args.dependency, args.prepare_env, args.smoke, args.jsonl))


def runs_static_validation(args: argparse.Namespace) -> bool:
    return args.static or not runs_dynamic_validation(args)


def select_examples(
    examples: Sequence[Mapping[str, object]],
    requested: Sequence[str],
    include_all: bool,
) -> List[Mapping[str, object]]:
    if include_all or not requested:
        return list(examples)

    requested_values = {normalize_selector(value) for value in requested}
    selected = []
    for example in examples:
        if example_matches_selectors(example, requested_values):
            selected.append(example)

    return selected


def example_matches_selectors(
    example: Mapping[str, object],
    selectors: Sequence[str],
) -> bool:
    values = (
        normalize_selector(str(example.get("name", ""))),
        normalize_selector(str(example.get("path", ""))),
        normalize_selector(str(example.get("benchmark_file", ""))),
    )
    return any(value in selectors for value in values)


def normalize_selector(value: str) -> str:
    value = value.strip().strip("\"'")
    if value.startswith("./"):
        value = value[2:]
    return value.rstrip("/")


def render_report(report, report_format: str) -> str:
    if report_format == REPORT_FORMAT_JSON:
        return render_json(report)
    return render_markdown(report)


def dependency_install_mode(args: argparse.Namespace) -> str:
    if args.pip_install:
        return INSTALL_MODE_INSTALL
    if args.pip_install_check:
        return INSTALL_MODE_DRY_RUN
    return INSTALL_MODE_SKIP


def active_examples(
    examples: Sequence[Mapping[str, object]],
) -> List[Mapping[str, object]]:
    return [
        example
        for example in examples
        if str(example.get("status", "active")) == "active"
    ]


def inactive_examples(
    examples: Sequence[Mapping[str, object]],
) -> List[Mapping[str, object]]:
    return [
        example
        for example in examples
        if str(example.get("status", "active")) != "active"
    ]


def skip_dynamic_examples(
    examples: Sequence[Mapping[str, object]],
) -> StaticValidationReport:
    reports = []
    for example in examples:
        example_path = normalize_selector(str(example.get("path", "")))
        example_name = str(example.get("name") or example_path)
        status = str(example.get("status", "unknown"))
        LOGGER.info(
            "Skipping dynamic validation for {} because inventory status is '{}'.".format(
                example_name,
                status,
            )
        )
        reports.append(
            ExampleReport(
                name=example_name,
                path=example_path,
                checks=[
                    CheckResult(
                        name="Dynamic validation eligibility",
                        status=SKIP,
                        message=(
                            "Example status is '{}'; dynamic validation is skipped "
                            "until the inventory marks it as active."
                        ).format(status),
                        file=str(example.get("benchmark_file") or example_path),
                        details=[
                            "inventory status: {}".format(status),
                            "example path: {}".format(example_path),
                        ],
                    )
                ],
            )
        )
    return StaticValidationReport(reports=reports)


def merge_reports(reports: Sequence[StaticValidationReport]) -> StaticValidationReport:
    merged_by_benchmark = {}
    for report in reports:
        for example_report in report.reports:
            key = (example_report.name, example_report.path)
            if key not in merged_by_benchmark:
                merged_by_benchmark[key] = example_report
                continue
            merged_by_benchmark[key].checks.extend(example_report.checks)

    return StaticValidationReport(
        reports=sorted(
            merged_by_benchmark.values(), key=lambda item: (item.path, item.name)
        )
    )


def write_or_print_report(rendered: str, report_path: str) -> None:
    if not report_path:
        sys.stdout.write(rendered)
        return

    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    LOGGER.info("Validation report written to %s", path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] - %(message)s")
    raise SystemExit(main())
