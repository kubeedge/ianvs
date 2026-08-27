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

"""Collect validator results and publish human-readable reports.

This script is intentionally dependency-free so GitHub Actions can run it before
project dependencies are installed. It accepts JSON reports emitted by
validation_runner.py, merges them, writes a Markdown report, emits GitHub
workflow annotations, and updates a pull request comment when running in a PR.
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import logging
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote, urlencode

try:
    from inventory_loader import DEFAULT_INVENTORY_PATH, load_inventory_examples
except ImportError:  # Support importing this file as services.report_generator.
    from services.inventory_loader import (
        DEFAULT_INVENTORY_PATH,
        load_inventory_examples,
    )


PASS = "PASS"
FAIL = "FAIL"
ERROR = "ERROR"
WARNING = "WARNING"
SKIP = "SKIP"
MODE_STATIC = "static"
MODE_DYNAMIC = "dynamic"
BLOCKING_STATUSES = {ERROR, FAIL}
DYNAMIC_ELIGIBILITY_CHECK = "Dynamic validation eligibility"
UNVALIDATED_REASON = "CI/CD ongoing"
ONGOING_REASON = "Example ongoing"
COMMENT_MARKER = "<!-- ianvs-example-validation-report -->"
MAX_COMMENT_BODY_CHARS = 60000
MAX_DYNAMIC_NEW_ERRORS = 10
MAX_DYNAMIC_NEW_WARNINGS = 10
RUNTIME_SMOKE_TEST_PREFIX = "Runtime smoke test"
DYNAMIC_EXCEPTION_TYPE_RE = re.compile(
    r"(?<![A-Za-z0-9_.])"
    r"(?P<error_type>[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception))(?=:)",
    re.MULTILINE,
)
DEFAULT_RESULT_PATTERNS = ("validation-results", "validator-results")
GITHUB_PULL_REQUEST_EVENTS = ("pull_request", "pull_request_target")
GITHUB_API_VERSION = "2022-11-28"
GITHUB_USER_AGENT = "ianvs-example-validator"
STATUS_RUNNABLE = "Runnable"
STATUS_BROKEN = "Broken"
STATUS_CICD_ONGOING = "CI/CD onGoing"
STATUS_EXAMPLE_ONGOING = "Example onGoing"
STATUS_EXTERNAL = "Requires external dataset or model download"
STATUS_HARDWARE = "Requires GPU or special hardware"
STATUS_QUARANTINED = "Quarantined"
STATUS_KNOWN = "Known issue"
REASON_DEPENDENCY = "Dependency drift"
REASON_RESOURCE = "Dataset or resource unavailable"
STATUS_COLORS = {
    STATUS_RUNNABLE: "brightgreen",
    STATUS_BROKEN: "red",
    STATUS_CICD_ONGOING: "lightgrey",
    STATUS_EXAMPLE_ONGOING: "yellow",
    STATUS_EXTERNAL: "blue",
    STATUS_HARDWARE: "orange",
    STATUS_QUARANTINED: "8a2be2",
    STATUS_KNOWN: "critical",
}
REASON_COLORS = {
    REASON_DEPENDENCY: "ff69b4",
    REASON_RESOURCE: "795548",
}
MANUAL_STATUS_BY_INVENTORY = {
    "quarantined": STATUS_QUARANTINED,
    "known issue": STATUS_KNOWN,
    "known_issue": STATUS_KNOWN,
    "hardware": STATUS_HARDWARE,
    "requires_hardware": STATUS_HARDWARE,
    "external": STATUS_EXTERNAL,
    "requires_external_resource": STATUS_EXTERNAL,
    "broken": STATUS_BROKEN,
}
SKIP_REASON_STATUSES = (
    STATUS_QUARANTINED,
    STATUS_KNOWN,
    STATUS_HARDWARE,
)
STATUS_REPOSITORY = "kubeedge/ianvs"
STATUS_BRANCH = "ci-managed/example-health-status"
STATUS_RESULT_ROOT = ".github/example-status"
LOGGER = logging.getLogger("ianvs.validator.report")


@dataclass
class CheckResult:
    name: str
    status: str
    message: str = ""
    file: str = ""
    details: List[str] = field(default_factory=list)

    @property
    def issue_count(self) -> int:
        if (
            self.status in BLOCKING_STATUSES
            and self.name.startswith(RUNTIME_SMOKE_TEST_PREFIX)
        ):
            return 1
        return len(self.details) if self.details else 1


@dataclass
class ExampleResult:
    name: str
    path: str
    passed: bool
    checks: List[CheckResult] = field(default_factory=list)

    @property
    def identity(self) -> Tuple[str, str]:
        return self.name, self.path.rstrip("/")

    def count(self, status: str) -> int:
        return sum(check.issue_count for check in self.checks if check.status == status)

    def count_any(self, statuses: Sequence[str]) -> int:
        return sum(
            check.issue_count for check in self.checks if check.status in statuses
        )

    @property
    def has_blocking_errors(self) -> bool:
        return any(check.status in BLOCKING_STATUSES for check in self.checks)


@dataclass
class CombinedReport:
    examples: List[ExampleResult]
    source_files: List[str]

    @property
    def passed(self) -> bool:
        return all(not example.has_blocking_errors for example in self.examples)

    def check_count(self, status: str) -> int:
        return sum(example.count(status) for example in self.examples)

    def check_count_any(self, statuses: Sequence[str]) -> int:
        return sum(example.count_any(statuses) for example in self.examples)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Ianvs validator JSON results and publish reports."
    )
    parser.add_argument(
        "--results",
        action="append",
        default=[],
        help="JSON result file, directory, or glob. Can be repeated.",
    )
    parser.add_argument(
        "--mode",
        choices=(MODE_STATIC, MODE_DYNAMIC),
        default=MODE_DYNAMIC,
        help=(
            "Report layout. Static includes ERROR and WARNING; "
            "dynamic keeps the error-only layout."
        ),
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional Markdown report output path.",
    )
    parser.add_argument(
        "--regression-json",
        default="",
        help="Optional regression detector JSON output to include as a summary section.",
    )
    parser.add_argument(
        "--artifacts-json",
        default="",
        help="Optional GitHub Actions run artifacts JSON used for runtime log links.",
    )
    parser.add_argument(
        "--step-summary",
        action="store_true",
        help="Append the Markdown report to GITHUB_STEP_SUMMARY when available.",
    )
    parser.add_argument(
        "--annotations",
        action="store_true",
        help="Emit GitHub Actions error annotations for failed checks.",
    )
    parser.add_argument(
        "--pr-comment",
        action="store_true",
        help="Create or update a pull request comment when running in a PR event.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Generate a passing empty report when no result files are found.",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Exit 0 even when collected validation results contain failures.",
    )
    parser.add_argument(
        "--example-health-readme",
        default="",
        help="Optional examples/README.md path to update from the collected results.",
    )
    parser.add_argument(
        "--inventory",
        default=DEFAULT_INVENTORY_PATH,
        help="Inventory used when rendering the example health README.",
    )
    parser.add_argument(
        "--health-metadata",
        default="",
        help="Optional JSON metadata containing T2/T3 source and validation time.",
    )
    parser.add_argument(
        "--example-status-output",
        default="",
        help="Optional directory for example-status snapshot JSON files.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result_paths = discover_result_paths(args.results)
    if not result_paths and not args.allow_empty:
        LOGGER.error("No validation result JSON files were found.")
        return 2

    report = load_combined_report(result_paths)
    inventory_path = Path(args.inventory)
    inventory_examples = (
        load_inventory_examples(inventory_path, active_only=False)
        if inventory_path.is_file()
        else []
    )
    validation_artifact_links = load_validation_artifact_links(
        args.artifacts_json,
        result_paths,
        mode=args.mode,
    )
    rendered = render_full_report(
        report,
        mode=args.mode,
        regression_json=args.regression_json,
        runtime_artifact_links=validation_artifact_links["pr"],
        base_artifact_links=validation_artifact_links["base"],
        inventory_examples=inventory_examples,
    )
    publish_report(rendered, report, args, mode=args.mode)
    if args.example_health_readme:
        inventory_examples = load_inventory_examples(
            Path(args.inventory), active_only=False
        )
        metadata = load_health_metadata(args.health_metadata)
        health_readme = render_example_health_readme(
            inventory_examples, report, metadata
        )
        write_text_file(health_readme, args.example_health_readme)
    if args.example_status_output:
        inventory_examples = load_inventory_examples(
            Path(args.inventory), active_only=False
        )
        metadata = load_health_metadata(args.health_metadata)
        snapshots = create_example_status_snapshots(
            report,
            inventory_examples,
            str(metadata["validated_at"]),
            str(metadata["source_sha"]),
        )
        if not snapshots:
            LOGGER.error("No validation results matched the example inventory.")
            return 2
        write_example_status_snapshots(
            snapshots,
            {
                "validated_at": str(metadata["validated_at"]),
                "validated_at_display": format_validation_time(
                    str(metadata["validated_at"])
                ),
                "commit": str(metadata["source_sha"]),
            },
            Path(args.example_status_output),
        )

    if report.passed or args.no_fail:
        return 0
    return 1


def discover_result_paths(inputs: Sequence[str]) -> List[Path]:
    patterns = list(inputs) if inputs else list(DEFAULT_RESULT_PATTERNS)
    paths = []

    for value in patterns:
        paths.extend(discover_json_paths(value))

    return unique_paths(paths)


def discover_json_paths(value: str) -> List[Path]:
    path = Path(value)
    if path.is_dir():
        return sorted(
            candidate for candidate in path.rglob("*.json") if candidate.is_file()
        )
    if path.is_file():
        return [path]

    matches = [Path(match) for match in glob.glob(value, recursive=True)]
    return sorted(
        match for match in matches if match.is_file() and match.suffix == ".json"
    )


def unique_paths(paths: Sequence[Path]) -> List[Path]:
    seen = set()
    unique_paths = []
    for path in paths:
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique_paths.append(path)
    return unique_paths


def load_combined_report(paths: Sequence[Path]) -> CombinedReport:
    examples = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        examples.extend(parse_examples(payload, source_path=path))

    examples = merge_duplicate_examples(examples)
    examples.sort(key=lambda example: (example.path, example.name))
    return CombinedReport(
        examples=examples,
        source_files=[path.as_posix() for path in paths],
    )


def load_validation_artifact_links(
    artifacts_json: str,
    result_paths: Sequence[Path],
    mode: str = MODE_DYNAMIC,
) -> Dict[str, Dict[str, str]]:
    links: Dict[str, Dict[str, str]] = {"base": {}, "pr": {}}
    artifacts_path = Path(artifacts_json)
    server_url = os.environ.get("GITHUB_SERVER_URL", "").rstrip("/")
    repository = os.environ.get("GITHUB_REPOSITORY", "").strip("/")
    run_id = os.environ.get("GITHUB_RUN_ID", "").strip()
    if (
        not artifacts_json
        or not artifacts_path.is_file()
        or not server_url
        or not repository
        or not run_id
    ):
        return links

    try:
        payload = json.loads(artifacts_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return links
    pages = payload if isinstance(payload, list) else [payload]
    artifact_urls = {}
    for page in pages:
        if not isinstance(page, dict):
            continue
        artifacts = page.get("artifacts", [])
        if not isinstance(artifacts, list):
            continue
        for artifact in artifacts:
            if not isinstance(artifact, dict) or artifact.get("expired"):
                continue
            name = str(artifact.get("name") or "")
            artifact_id = str(artifact.get("id") or "")
            if not name or not artifact_id:
                continue
            artifact_urls[name] = "{}/{}/actions/runs/{}/artifacts/{}".format(
                server_url,
                repository,
                run_id,
                artifact_id,
            )

    for result_path in result_paths:
        result_artifact_urls = {
            revision: artifact_urls.get(
                "{}-validation-{}-{}".format(
                    mode, revision, result_path.stem
                )
            )
            for revision in ("base", "pr")
        }
        if not any(result_artifact_urls.values()):
            continue
        try:
            result_payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        raw_examples = result_payload.get("examples", [])
        if not isinstance(raw_examples, list):
            continue
        for raw_example in raw_examples:
            if not isinstance(raw_example, dict):
                continue
            example_path = str(
                raw_example.get("path") or raw_example.get("name") or ""
            )
            example_name = str(
                raw_example.get("name") or raw_example.get("path") or ""
            )
            if example_path:
                for revision, artifact_url in result_artifact_urls.items():
                    if artifact_url:
                        links[revision][
                            example_artifact_key(example_name, example_path)
                        ] = artifact_url
                        links[revision].setdefault(example_path, artifact_url)
    return links


def load_runtime_artifact_links(
    artifacts_json: str,
    result_paths: Sequence[Path],
) -> Dict[str, str]:
    """Return PR artifact links for callers using the original helper."""
    return load_validation_artifact_links(artifacts_json, result_paths)["pr"]


def merge_duplicate_examples(examples: Sequence[ExampleResult]) -> List[ExampleResult]:
    merged: Dict[Tuple[str, str], ExampleResult] = {}

    for example in examples:
        key = example.identity
        if key not in merged:
            merged[key] = ExampleResult(
                name=example.name,
                path=example.path,
                passed=example.passed,
                checks=[copy_check(check) for check in example.checks],
            )
            continue

        target = merged[key]
        target.passed = target.passed and example.passed
        target.checks = merge_checks([*target.checks, *example.checks])
        target.passed = not target.has_blocking_errors

    return list(merged.values())


def merge_checks(checks: Sequence[CheckResult]) -> List[CheckResult]:
    merged: Dict[Tuple[str, str, str, str], CheckResult] = {}

    for check in checks:
        key = (check.name, check.status, check.message, check.file)
        if key not in merged:
            merged[key] = copy_check(check)
            continue

        target = merged[key]
        seen_details = set(target.details)
        for detail in check.details:
            if detail in seen_details:
                continue
            seen_details.add(detail)
            target.details.append(detail)

    return sorted(
        merged.values(),
        key=lambda check: (check.name, check.status, check.file, check.message),
    )


def copy_check(check: CheckResult) -> CheckResult:
    return CheckResult(
        name=check.name,
        status=check.status,
        message=check.message,
        file=check.file,
        details=list(check.details),
    )


def parse_examples(payload: Dict[str, object], source_path: Path) -> List[ExampleResult]:
    raw_examples = payload.get("examples")
    if not isinstance(raw_examples, list):
        raise ValueError("{} does not contain an examples list".format(source_path))

    examples = []
    for raw_example in raw_examples:
        if not isinstance(raw_example, dict):
            continue

        checks = parse_checks(raw_example.get("checks", []))
        examples.append(
            ExampleResult(
                name=str(raw_example.get("name") or raw_example.get("path") or ""),
                path=str(raw_example.get("path") or raw_example.get("name") or ""),
                passed=not any(check.status in BLOCKING_STATUSES for check in checks),
                checks=checks,
            )
        )
    return examples


def parse_checks(raw_checks: object) -> List[CheckResult]:
    if not isinstance(raw_checks, list):
        return []

    checks = []
    for raw_check in raw_checks:
        if not isinstance(raw_check, dict):
            continue
        checks.append(
            CheckResult(
                name=str(raw_check.get("name", "")),
                status=str(raw_check.get("status", "")),
                message=str(raw_check.get("message", "")),
                file=str(raw_check.get("file", "")),
                details=string_list(raw_check.get("details", [])),
            )
        )
    return checks


def string_list(value: object) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def render_full_report(
    report: CombinedReport,
    mode: str = MODE_DYNAMIC,
    regression_json: str = "",
    runtime_artifact_links: Optional[Dict[str, str]] = None,
    base_artifact_links: Optional[Dict[str, str]] = None,
    inventory_examples: Sequence[dict] = (),
) -> str:
    if mode == MODE_DYNAMIC:
        rendered = render_dynamic_markdown(report, include_skipped_examples=False)
    else:
        rendered = render_markdown(report, mode=mode)
    if regression_json:
        rendered = append_regression_summary(
            rendered,
            Path(regression_json),
            mode=mode,
            excluded_examples=dynamic_skipped_example_paths(report)
            if mode == MODE_DYNAMIC
            else (),
            runtime_artifact_links=runtime_artifact_links,
            base_artifact_links=base_artifact_links,
        )
    rendered = append_collected_result_files(
        rendered,
        report,
        base_artifact_links=base_artifact_links,
        pr_artifact_links=runtime_artifact_links,
        inventory_examples=inventory_examples,
    )
    if mode == MODE_DYNAMIC:
        rendered = append_dynamic_skipped_examples(rendered, report)
    return rendered


def publish_report(
    rendered: str,
    report: CombinedReport,
    args: argparse.Namespace,
    mode: str = MODE_DYNAMIC,
) -> None:
    write_or_print_report(rendered, args.output)

    if args.step_summary:
        append_step_summary(rendered)

    if args.annotations:
        emit_annotations(report, mode=mode)

    if args.pr_comment:
        maybe_update_pr_comment(rendered)


def write_or_print_report(rendered: str, output: str) -> None:
    if not output:
        sys.stdout.write(rendered)
        return

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    LOGGER.info("Combined validation report written to %s", output_path)


def write_text_file(rendered: str, output: str) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    LOGGER.info("Example health report written to %s", output_path)


def load_health_metadata(path: str) -> Dict[str, object]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Health metadata must be a JSON object")
    return payload


def parse_utc(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_validation_time(value: str) -> str:
    return parse_utc(value).strftime("%Y-%m-%d %H:%M UTC")


def render_example_health_readme(
    inventory_examples: Sequence[dict],
    report: CombinedReport,
    metadata: Dict[str, object],
) -> str:
    lines = [
        "# Ianvs Examples",
        "",
        (
            "For status meanings, badge definitions, and broken-status subtypes, "
            "see [`status_directions.md`](../docs/proposals/scenarios/"
            "example-restoration/phase-3-2026-term-2/status_directions.md)."
        ),
        "",
    ]
    lines.append(
        "**Last T2/T3 Validation Time:** {}".format(
            dynamic_json_badge(
                "validated at", "summary.json", "$.validated_at_display"
            )
        )
    )

    lines.extend(
        [
            "",
            "## Example Classification Matrix",
            "",
            "<table>",
            "  <thead>",
            "    <tr>",
            "      <th>Example</th>",
            "      <th>Benchmark Unit</th>",
            "      <th>Status</th>",
            "    </tr>",
            "  </thead>",
            "  <tbody>",
        ]
    )
    grouped_examples: Dict[str, List[dict]] = {}
    for inventory_example in inventory_examples:
        example_name = str(
            inventory_example.get("example")
            or inventory_example.get("name")
            or inventory_example.get("path", "")
        )
        grouped_examples.setdefault(example_name, []).append(inventory_example)

    for example_name in sorted(grouped_examples):
        benchmark_units = sorted(
            grouped_examples[example_name],
            key=lambda item: (
                str(item.get("name", "")),
                str(item.get("benchmark_file", "")),
                str(item.get("path", "")),
            ),
        )
        for index, inventory_example in enumerate(benchmark_units):
            path = str(inventory_example.get("path", "")).rstrip("/")
            benchmark_name = str(inventory_example.get("name") or path)
            readme_path = path[9:] if path.startswith("examples/") else path
            benchmark_link = '<a href="./{}">{}</a>'.format(
                quote(readme_path, safe="/"),
                html.escape(benchmark_name),
            )
            lines.append("    <tr>")
            if index == 0:
                if len(benchmark_units) > 1:
                    lines.append(
                        '      <td rowspan="{}">{}</td>'.format(
                            len(benchmark_units), html.escape(example_name)
                        )
                    )
                else:
                    lines.append(
                        "      <td>{}</td>".format(html.escape(example_name))
                    )
            lines.extend(
                [
                    "      <td>{}</td>".format(benchmark_link),
                    "      <td>{}</td>".format(
                        endpoint_json_badge(
                            "status", status_file_name(example_name)
                        )
                    ),
                    "    </tr>",
                ]
            )
    lines.extend(["  </tbody>", "</table>"])
    return "\n".join(lines).rstrip() + "\n"


def status_file_name(example: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", example.strip())
    normalized = normalized.strip("._")
    if not normalized:
        raise ValueError("Example name does not contain a safe filename character")
    return normalized + ".json"


def create_example_status_snapshots(
    report: CombinedReport,
    inventory_examples: Sequence[dict],
    validated_at: str,
    commit: str,
) -> Dict[str, dict]:
    result_by_identity = {example.identity: example for example in report.examples}
    grouped_inventory: Dict[str, List[dict]] = {}
    grouped_results: Dict[str, List[ExampleResult]] = {}

    for inventory_example in inventory_examples:
        example = str(
            inventory_example.get("example")
            or inventory_example.get("name")
            or inventory_example.get("path", "")
        )
        grouped_inventory.setdefault(example, []).append(inventory_example)
        name = str(inventory_example.get("name", ""))
        path = str(inventory_example.get("path", "")).rstrip("/")
        result = result_by_identity.get((name, path))
        if result is not None:
            grouped_results.setdefault(example, []).append(result)

    snapshots = {}
    for example, inventory_group in grouped_inventory.items():
        results = grouped_results.get(example, [])
        has_failure = any(result.has_blocking_errors for result in results)
        inventory_statuses = {
            str(item.get("status", "active")).lower() for item in inventory_group
        }
        manual_statuses = {
            MANUAL_STATUS_BY_INVENTORY.get(inventory_status)
            for inventory_status in inventory_statuses
        }
        skip_reason = next(
            (
                status
                for status in SKIP_REASON_STATUSES
                if status in manual_statuses
            ),
            "",
        )
        if has_failure:
            status = "failing"
            message = STATUS_BROKEN
            label = "status"
        elif skip_reason:
            status = "skipped"
            message = skip_reason
            label = "reason"
        elif "ongoing" in inventory_statuses:
            status = STATUS_EXAMPLE_ONGOING
            message = STATUS_EXAMPLE_ONGOING
            label = "status"
        elif "unvalidated" in inventory_statuses or not results:
            status = STATUS_CICD_ONGOING
            message = STATUS_CICD_ONGOING
            label = "status"
        else:
            status = "passing"
            message = STATUS_RUNNABLE
            label = "status"
        snapshots[status_file_name(example)] = {
            "example": example,
            "status": status,
            "validated_at": validated_at,
            "commit": commit,
            "schemaVersion": 1,
            "label": label,
            "message": message,
            "color": STATUS_COLORS[message],
        }
    return snapshots


def write_example_status_snapshots(
    snapshots: Dict[str, dict], summary: dict, output: Path
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for filename, payload in sorted(snapshots.items()):
        (output / filename).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def dynamic_json_badge(label: str, filename: str, query: str) -> str:
    raw_url = "https://raw.githubusercontent.com/{}/{}/{}/{}".format(
        STATUS_REPOSITORY, STATUS_BRANCH, STATUS_RESULT_ROOT, filename
    )
    badge_url = "https://img.shields.io/badge/dynamic/json?{}".format(
        urlencode(
            {
                "url": raw_url,
                "query": query,
                "label": label,
                "cacheSeconds": "300",
            }
        )
    )
    return '<img alt="{}" src="{}">'.format(
        html.escape(label, quote=True), html.escape(badge_url, quote=True)
    )


def endpoint_json_badge(label: str, filename: str) -> str:
    raw_url = "https://raw.githubusercontent.com/{}/{}/{}/{}".format(
        STATUS_REPOSITORY, STATUS_BRANCH, STATUS_RESULT_ROOT, filename
    )
    badge_url = "https://img.shields.io/endpoint?{}".format(
        urlencode({"url": raw_url, "cacheSeconds": "300"})
    )
    return '<img alt="{}" src="{}">'.format(
        html.escape(label, quote=True), html.escape(badge_url, quote=True)
    )


def classify_health_status(
    inventory_example: dict,
    result: Optional[ExampleResult],
) -> Tuple[str, str]:
    inventory_status = str(inventory_example.get("status", "unvalidated")).lower()
    if inventory_status in MANUAL_STATUS_BY_INVENTORY:
        return MANUAL_STATUS_BY_INVENTORY[inventory_status], ""
    if inventory_status == "unvalidated":
        return STATUS_CICD_ONGOING, ""
    if inventory_status == "ongoing":
        return STATUS_EXAMPLE_ONGOING, ""
    if inventory_status != "active" or result is None:
        return STATUS_CICD_ONGOING, ""

    failed_checks = [
        check for check in result.checks if check.status in BLOCKING_STATUSES
    ]
    if not failed_checks:
        return STATUS_RUNNABLE, ""
    failure_text = " ".join(
        " ".join([check.name, check.message, check.file, " ".join(check.details)])
        .lower()
        for check in failed_checks
    )
    if any(
        word in failure_text
        for word in ("dependency", "requirements", "package", "pip")
    ):
        return STATUS_BROKEN, REASON_DEPENDENCY
    if any(
        word in failure_text
        for word in ("dataset", "jsonl", "model", "resource", "download")
    ):
        return STATUS_BROKEN, REASON_RESOURCE
    return STATUS_BROKEN, ""


def render_health_badges(status: str, reason: str) -> str:
    rendered = health_badge("status", status, STATUS_COLORS[status])
    if reason:
        rendered += " " + health_badge("reason", reason, REASON_COLORS[reason])
    return rendered


def health_badge(label: str, value: str, color: str) -> str:
    image_url = "https://img.shields.io/badge/{}-{}-{}".format(
        quote(label, safe=""), quote(value, safe=""), color
    )
    return '<img alt="{}" src="{}">'.format(
        html.escape(value, quote=True),
        html.escape(image_url, quote=True),
    )


def render_markdown(
    report: CombinedReport,
    mode: str = MODE_DYNAMIC,
) -> str:
    if mode == MODE_STATIC:
        return render_static_markdown(report)
    return render_dynamic_markdown(report)


def render_static_markdown(report: CombinedReport) -> str:
    result = static_overall_result(report)
    lines = [
        COMMENT_MARKER,
        "# Ianvs Static Validation Report",
        "",
        "**Overall result:** {}".format(result),
        "",
        "| Examples | Errors | Warnings | Skipped checks |",
        "|---:|---:|---:|---:|",
        "| {} | {} | {} | {} |".format(
            len(report.examples),
            report.check_count_any(BLOCKING_STATUSES),
            report.check_count(WARNING),
            report.check_count(SKIP),
        ),
        "",
    ]

    if not report.examples:
        lines.append("No validation results were collected.")
        return "\n".join(lines).rstrip() + "\n"

    lines.extend(
        [
            "## Example Summary",
            "",
            "| Example | Result | Errors | Warnings | Skip |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for example in report.examples:
        lines.append(
            "| `{}` | {} | {} | {} | {} |".format(
                escape_table(example.path),
                static_example_result(example),
                example.count_any(BLOCKING_STATUSES),
                example.count(WARNING),
                example.count(SKIP),
            )
        )

    return "\n".join(lines).rstrip() + "\n"


def render_dynamic_markdown(
    report: CombinedReport,
    include_skipped_examples: bool = True,
) -> str:
    skipped_examples = dynamic_skipped_examples(report)
    runnable_examples = [
        example for example in report.examples if example.path not in skipped_examples
    ]
    result = PASS if all(
        not example.has_blocking_errors for example in runnable_examples
    ) else FAIL
    lines = [
        COMMENT_MARKER,
        "# Ianvs Dynamic Validation Report",
        "",
        "**Overall result:** {}".format(result),
        "",
        "| Selected | Executed | Skipped | Errors |",
        "|---:|---:|---:|---:|",
        "| {} | {} | {} | {} |".format(
            len(report.examples),
            len(runnable_examples),
            len(skipped_examples),
            sum(
                example.count_any(BLOCKING_STATUSES)
                for example in runnable_examples
            ),
        ),
        "",
    ]

    if not runnable_examples and not skipped_examples:
        lines.append("No validation results were collected.")
        return "\n".join(lines).rstrip() + "\n"

    if runnable_examples:
        lines.extend(
            [
                "## Example Summary",
                "",
                "| Example | Result | Errors | Skip |",
                "|---|---:|---:|---:|",
            ]
        )
        for example in runnable_examples:
            lines.append(
                "| `{}` | {} | {} | {} |".format(
                    escape_table(example.path),
                    dynamic_example_result(example),
                    example.count_any(BLOCKING_STATUSES),
                    example.count(SKIP),
                )
            )

    rendered = "\n".join(lines).rstrip() + "\n"
    if include_skipped_examples:
        return append_dynamic_skipped_examples(rendered, report)
    return rendered


def append_dynamic_skipped_examples(
    rendered: str,
    report: CombinedReport,
) -> str:
    skipped_examples = dynamic_skipped_examples(report)
    if not skipped_examples:
        return rendered

    lines = [
        "",
        "<details>",
        "<summary><h2>Skipped Examples</h2></summary>",
        "",
        "| Example | Reason |",
        "|---|---|",
    ]
    for example_path, reason in sorted(skipped_examples.items()):
        lines.append("| `{}` | {} |".format(escape_table(example_path), reason))
    lines.extend(["", "</details>"])
    return rendered.rstrip() + "\n" + "\n".join(lines).rstrip() + "\n"


def dynamic_skipped_examples(report: CombinedReport) -> Dict[str, str]:
    skipped = {}
    for example in report.examples:
        reason = dynamic_skip_reason(example)
        if reason:
            skipped[example.path] = reason
    return skipped


def dynamic_skipped_example_paths(report: CombinedReport) -> List[str]:
    return sorted(dynamic_skipped_examples(report))


def dynamic_skip_reason(example: ExampleResult) -> str:
    for check in example.checks:
        if check.name != DYNAMIC_ELIGIBILITY_CHECK or check.status != SKIP:
            continue
        inventory_status = inventory_status_from_check(check)
        if inventory_status == "unvalidated":
            return UNVALIDATED_REASON
        if inventory_status == "ongoing":
            return ONGOING_REASON
        manual_status = MANUAL_STATUS_BY_INVENTORY.get(inventory_status)
        if manual_status in SKIP_REASON_STATUSES:
            return health_badge(
                "reason", manual_status, STATUS_COLORS[manual_status]
            )
    return ""


def inventory_status_from_check(check: CheckResult) -> str:
    for detail in check.details:
        prefix = "inventory status:"
        if detail.lower().startswith(prefix):
            return detail[len(prefix):].strip().lower()
    return ""


def static_overall_result(report: CombinedReport) -> str:
    if report.check_count_any(BLOCKING_STATUSES):
        return ERROR
    if report.check_count(WARNING):
        return WARNING
    return PASS


def static_example_result(example: ExampleResult) -> str:
    if example.count_any(BLOCKING_STATUSES):
        return ERROR
    if example.count(WARNING):
        return WARNING
    if example.count(SKIP):
        return SKIP
    return PASS


def dynamic_example_result(example: ExampleResult) -> str:
    if example.count_any(BLOCKING_STATUSES):
        return FAIL
    if example.count(SKIP):
        return SKIP
    return PASS


def append_regression_summary(
    rendered: str,
    regression_json_path: Path,
    mode: str = MODE_DYNAMIC,
    excluded_examples: Sequence[str] = (),
    runtime_artifact_links: Optional[Dict[str, str]] = None,
    base_artifact_links: Optional[Dict[str, str]] = None,
) -> str:
    if not regression_json_path.is_file():
        return rendered

    payload = json.loads(regression_json_path.read_text(encoding="utf-8"))
    comparisons = payload.get("comparisons", [])
    if not isinstance(comparisons, list):
        comparisons = []
    example_changes = payload.get("example_changes", [])
    if not isinstance(example_changes, list):
        example_changes = []
    excluded = set(excluded_examples)
    comparisons = [
        comparison
        for comparison in comparisons
        if not isinstance(comparison, dict)
        or str(comparison.get("example") or "") not in excluded
    ]

    examples = regression_examples(comparisons)
    if mode == MODE_STATIC:
        summary = static_regression_summary(comparisons, examples)
    else:
        summary = dynamic_regression_summary(
            comparisons,
            examples,
            runtime_artifact_links=runtime_artifact_links,
            base_artifact_links=base_artifact_links,
        )
    summary = regression_example_change_summary(example_changes) + summary
    summary.append("")
    return rendered.rstrip() + "\n" + "\n".join(summary).rstrip() + "\n"


def regression_example_change_summary(changes: Sequence[object]) -> List[str]:
    normalized = [change for change in changes if isinstance(change, dict)]
    if not normalized:
        return ["", "**Example changes:** None", ""]

    added_count = sum(change.get("change") == "Added" for change in normalized)
    removed_count = sum(change.get("change") == "Removed" for change in normalized)
    summary = [
        "",
        "## Example Changes",
        "",
        "- Added examples: {}".format(added_count),
        "- Removed examples: {}".format(removed_count),
        "",
        "| Change | Example | Validation | Classification | Blocks PR |",
        "|---|---|---|---|---|",
    ]
    for change in normalized:
        result = str(change.get("validation") or "Unknown")
        previous_state = str(change.get("previous_validation_state") or "")
        if change.get("change") == "Removed" and previous_state:
            result = "Removed (base: {})".format(previous_state)
        inventory_status = str(change.get("inventory_status") or "")
        if inventory_status:
            result = "{} (`{}`)".format(result, inventory_status)
        path = str(change.get("path") or "")
        name = str(change.get("name") or path)
        summary.append(
            "| {} | `{}` (`{}`) | {} | {} | {} |".format(
                escape_table(str(change.get("change") or "")),
                escape_table(path),
                escape_table(name),
                escape_table(result),
                escape_table(str(change.get("classification") or "")),
                "Yes" if change.get("blocks_pr") else "No",
            )
        )
    summary.append("")
    return summary


def static_regression_summary(
    comparisons: Sequence[object],
    examples: Sequence[str],
) -> List[str]:
    summary = [
        "",
        "## Regression Summary",
        "",
        (
            "Compares the base branch and PR validation results to identify new failures. "
            "Pre-existing failures do not block validation."
        ),
        "",
        "### ERROR",
        "",
        "| Example | Current errors | Pre-existing errors | New errors | Fixed errors |",
        "|---|---:|---:|---:|---:|",
    ]
    for example in examples:
        summary.append(regression_error_summary_row(comparisons, example))
    if not examples:
        summary.append("| No regression comparisons were collected. | 0 | 0 | 0 | 0 |")

    append_static_regression_details(
        summary,
        comparisons,
        severity=ERROR,
    )

    summary.extend(
        [
            "",
            "### Warnings",
            "",
            "| Example | Current warnings | Pre-existing warnings | New warnings | Fixed warnings |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for example in examples:
        summary.append(regression_warning_summary_row(comparisons, example))
    if not examples:
        summary.append("| No regression comparisons were collected. | 0 | 0 | 0 | 0 |")
    append_static_regression_details(
        summary,
        comparisons,
        severity=WARNING,
    )
    return summary


def append_static_regression_details(
    summary: List[str],
    comparisons: Sequence[object],
    severity: str,
) -> None:
    label = "ERRORs" if severity == ERROR else "warnings"
    for issue_kind in ("New", "Pre-existing"):
        rows = static_regression_problem_rows(comparisons, severity, issue_kind)
        if not rows:
            continue
        summary.extend(
            [
                "",
                "<details>",
                "<summary><h4>{} {}</h4></summary>".format(issue_kind, label),
                "",
                "| Example | Severity | Check | Problem and impact |",
                "|---|---:|---|---|",
            ]
        )
        for example, check, problem_and_impact in rows:
            summary.append(
                "| `{}` | {} | `{}` | {} |".format(
                    escape_table(example),
                    severity,
                    escape_table(check),
                    escape_table(problem_and_impact),
                )
            )
        summary.extend(["", "</details>"])


def static_regression_problem_rows(
    comparisons: Sequence[object],
    severity: str,
    issue_kind: str,
) -> List[Tuple[str, str, str]]:
    is_error = severity == ERROR
    if issue_kind == "New":
        count_field = "new_issue_count" if is_error else "new_warning_count"
        fallback_classification = "Failed: PR regression" if is_error else ""
    else:
        count_field = (
            "pre_existing_issue_count"
            if is_error
            else "pre_existing_warning_count"
        )
        fallback_classification = (
            "Failed: Pre-existing failure" if is_error else ""
        )

    rows = []
    seen = set()
    for comparison in comparisons:
        if not isinstance(comparison, dict):
            continue
        if not count_regression_field(
            [comparison],
            count_field,
            fallback_classification=fallback_classification,
        ):
            continue
        example = str(comparison.get("example") or "")
        check = str(comparison.get("check") or "")
        identity = (example, check)
        if identity in seen:
            continue
        seen.add(identity)
        problem_and_impact = str(comparison.get("message") or "").strip()
        if not problem_and_impact:
            problem_and_impact = "See the validation check output for impact details."
        rows.append((example, check, problem_and_impact))
    return rows


def dynamic_regression_summary(
    comparisons: Sequence[object],
    examples: Sequence[str],
    runtime_artifact_links: Optional[Dict[str, str]] = None,
    base_artifact_links: Optional[Dict[str, str]] = None,
) -> List[str]:
    new_error_count = count_regression_field(
        comparisons,
        "new_issue_count",
        fallback_classification="Failed: PR regression",
    )
    summary = [
        "",
        "## Regression Summary",
        "",
        (
            "Compares the base branch and PR validation results to identify new failures. "
            "Pre-existing failures do not block validation."
        ),
        "",
        "**Result:** {} — {}".format(
            FAIL if new_error_count else PASS,
            "New ERRORs were detected."
            if new_error_count
            else "No new ERRORs were detected.",
        ),
        "",
        "| Example | Current errors | Pre-existing errors | New errors | Fixed errors |",
        "|---|---:|---:|---:|---:|",
    ]
    for example in examples:
        summary.append(regression_error_summary_row(comparisons, example))
    if not examples:
        summary.append("| No regression comparisons were collected. | 0 | 0 | 0 | 0 |")

    for error_kind in ("New", "Pre-existing"):
        errors = regression_issue_group_details(comparisons, ERROR, error_kind)
        if not errors:
            continue
        summary.extend(
            [
                "",
                "<details>",
                "<summary><h3>{} ERRORs</h3></summary>".format(error_kind),
                "",
                "| Example | Check | ERROR |",
                "|---|---|---|",
            ]
        )
        error_artifact_links = (
            runtime_artifact_links if error_kind == "New" else base_artifact_links
        )
        for example, check, detail in errors:
            summary.append(
                "| `{}` | `{}` | {} |".format(
                    escape_table(example),
                    escape_table(check),
                    escape_table(
                        dynamic_error_detail(
                            example,
                            check,
                            detail,
                            runtime_artifact_links=error_artifact_links,
                        )
                    ),
                )
            )
        summary.extend(["", "</details>"])
    for warning_kind in ("New", "Pre-existing"):
        warnings = regression_issue_group_details(
            comparisons, WARNING, warning_kind
        )
        if not warnings:
            continue
        summary.extend(
            [
                "",
                "<details>",
                "<summary><h3>{} warnings</h3></summary>".format(warning_kind),
                "",
                "| Example | Check | Warning |",
                "|---|---|---|",
            ]
        )
        for example, check, detail in warnings:
            summary.append(
                "| `{}` | `{}` | {} |".format(
                    escape_table(example),
                    escape_table(check),
                    escape_table(detail),
                )
            )
        summary.extend(["", "</details>"])
    return summary


def regression_issue_group_details(
    comparisons: Sequence[object],
    severity: str,
    issue_kind: str,
) -> List[Tuple[str, str, str]]:
    is_error = severity == ERROR
    if issue_kind == "New":
        count_field = "new_issue_count" if is_error else "new_warning_count"
        details_field = "new_details" if is_error else "new_warning_details"
        fallback_classification = "Failed: PR regression" if is_error else ""
    else:
        count_field = (
            "pre_existing_issue_count"
            if is_error
            else "pre_existing_warning_count"
        )
        details_field = (
            "pre_existing_details"
            if is_error
            else "pre_existing_warning_details"
        )
        fallback_classification = (
            "Failed: Pre-existing failure" if is_error else ""
        )

    return regression_issue_details(
        comparisons,
        count_field=count_field,
        details_field=details_field,
        limit=None,
        fallback_classification=fallback_classification,
        fallback_details_field="details" if is_error else "",
        default_message="{} {} detected.".format(
            issue_kind, "ERROR" if is_error else "warning"
        ),
    )


def regression_issue_details(
    comparisons: Sequence[object],
    count_field: str,
    details_field: str,
    limit: Optional[int],
    fallback_classification: str = "",
    fallback_details_field: str = "",
    default_message: str = "Issue detected.",
) -> List[Tuple[str, str, str]]:
    issues = []
    for comparison in comparisons:
        if not isinstance(comparison, dict):
            continue
        issue_count = count_regression_field(
            [comparison],
            count_field,
            fallback_classification=fallback_classification,
        )
        if not issue_count:
            continue

        raw_details = comparison.get(details_field)
        if (
            (not isinstance(raw_details, list) or not raw_details)
            and fallback_details_field
        ):
            raw_details = comparison.get(fallback_details_field)
        details = string_list(raw_details)
        if not details:
            details = [str(comparison.get("message") or default_message)]

        example = str(comparison.get("example") or "")
        check = str(comparison.get("check") or "")
        for detail in details[:issue_count]:
            issues.append((example, check, detail))
            if limit is not None and len(issues) == limit:
                return issues
    return issues


def dynamic_new_errors(
    comparisons: Sequence[object],
    limit: int,
) -> List[Tuple[str, str, str]]:
    return regression_issue_details(
        comparisons,
        count_field="new_issue_count",
        details_field="new_details",
        limit=limit,
        fallback_classification="Failed: PR regression",
        fallback_details_field="details",
        default_message="New ERROR detected.",
    )


def dynamic_error_detail(
    example: str,
    check: str,
    detail: str,
    runtime_artifact_links: Optional[Dict[str, str]] = None,
) -> str:
    problem = dynamic_error_problem(check, detail)
    if not problem.endswith((".", "!", "?")):
        problem += "."
    artifact_url = (runtime_artifact_links or {}).get(example)
    target_url = artifact_url or github_actions_run_url()
    if not target_url:
        return "{} View execution logs.".format(problem)
    return "{} [View execution logs]({})".format(
        problem,
        target_url,
    )


def dynamic_error_problem(check: str, detail: str) -> str:
    problem = detail.strip()
    exception_match = DYNAMIC_EXCEPTION_TYPE_RE.search(problem)
    if check.startswith(RUNTIME_SMOKE_TEST_PREFIX) and exception_match:
        return "Example execution raise a `{}`".format(
            exception_match.group("error_type")
        )
    if check == "Smoke benchmark config exists":
        return "Can't find the benchmark file{}".format(
            ": `{}`".format(problem) if problem else ""
        )
    if check == "Dependency file exists":
        return "Can't find the dependency file{}".format(
            ": `{}`".format(problem) if problem else ""
        )
    if check == "Dataset preparation" and problem:
        return "Can't find the dataset preparation script: `{}`".format(problem)
    if check == "JSONL dataset structure" and problem.endswith(": file is missing"):
        return "Can't find the dataset file: `{}`".format(
            problem[: -len(": file is missing")]
        )

    return generic_dynamic_error_problem(check, detail)


def generic_dynamic_error_problem(check: str, detail: str) -> str:
    state = (
        "validator execution"
        if check == "Validation runner internal error"
        else check.strip() or "dynamic validation"
    )
    return "Workflow raises a `{}` during `{}`".format(
        dynamic_error_type(check, detail),
        state,
    )


def dynamic_error_type(check: str, detail: str) -> str:
    match = DYNAMIC_EXCEPTION_TYPE_RE.search(detail)
    if match:
        return match.group("error_type")

    if check.startswith(RUNTIME_SMOKE_TEST_PREFIX):
        return "runtime error"

    normalized_check = check.strip() or "dynamic validation"
    if normalized_check.lower().endswith(("error", "failure")):
        return normalized_check
    return "{} error".format(normalized_check)


def github_actions_run_url() -> str:
    server_url = os.environ.get("GITHUB_SERVER_URL", "").rstrip("/")
    repository = os.environ.get("GITHUB_REPOSITORY", "").strip("/")
    run_id = os.environ.get("GITHUB_RUN_ID", "").strip()
    if not server_url or not repository or not run_id:
        return ""
    return "{}/{}/actions/runs/{}".format(server_url, repository, run_id)


def dynamic_new_warnings(
    comparisons: Sequence[object],
    limit: int,
) -> List[Tuple[str, str, str]]:
    return regression_issue_details(
        comparisons,
        count_field="new_warning_count",
        details_field="new_warning_details",
        limit=limit,
        default_message="New warning detected.",
    )


def regression_error_summary_row(
    comparisons: Sequence[object],
    example: str,
) -> str:
    current_errors = count_regression_field(
        comparisons,
        "head_issue_count",
        example=example,
    )
    pre_existing_errors = count_regression_field(
        comparisons,
        "pre_existing_issue_count",
        fallback_classification="Failed: Pre-existing failure",
        example=example,
    )
    new_errors = count_regression_field(
        comparisons,
        "new_issue_count",
        fallback_classification="Failed: PR regression",
        example=example,
    )
    fixed_errors = count_regression_field(
        comparisons,
        "fixed_issue_count",
        fallback_classification="Fixed: Pre-existing failure resolved",
        example=example,
    )
    return "| `{}` | {} | {} | {} | {} |".format(
        escape_table(example),
        current_errors,
        pre_existing_errors,
        new_errors,
        fixed_errors,
    )


def regression_warning_summary_row(
    comparisons: Sequence[object],
    example: str,
) -> str:
    current_warnings = count_regression_field(
        comparisons,
        "head_warning_count",
        example=example,
    )
    pre_existing_warnings = count_regression_field(
        comparisons,
        "pre_existing_warning_count",
        example=example,
    )
    new_warnings = count_regression_field(
        comparisons,
        "new_warning_count",
        example=example,
    )
    fixed_warnings = count_regression_field(
        comparisons,
        "fixed_warning_count",
        example=example,
    )
    return "| `{}` | {} | {} | {} | {} |".format(
        escape_table(example),
        current_warnings,
        pre_existing_warnings,
        new_warnings,
        fixed_warnings,
    )


def append_collected_result_files(
    rendered: str,
    report: CombinedReport,
    base_artifact_links: Optional[Dict[str, str]] = None,
    pr_artifact_links: Optional[Dict[str, str]] = None,
    inventory_examples: Sequence[dict] = (),
) -> str:
    if not report.source_files:
        return rendered

    lines = [
        "",
        "<details>",
        "<summary><h2>Collected Result Files</h2></summary>",
        "",
    ]
    base_links = base_artifact_links or {}
    pr_links = pr_artifact_links or {}
    inventory_by_identity = {
        (
            str(item.get("name") or ""),
            str(item.get("path") or "").rstrip("/"),
        ): item
        for item in inventory_examples
    }
    if base_links or pr_links:
        lines.extend(
            [
                (
                    "The artifacts contain the validation result JSON and execution "
                    "log for each checked Example benchmark unit."
                ),
                "",
                "| Example | Benchmark Unit | Base Artifact | PR Artifact |",
                "|---|---|---|---|",
            ]
        )
        for example in report.examples:
            lines.append(
                "| {} | {} | {} | {} |".format(
                    example_folder_markdown(
                        example,
                        inventory_by_identity.get(example.identity),
                    ),
                    benchmark_file_markdown(
                        example,
                        inventory_by_identity.get(example.identity),
                    ),
                    artifact_markdown(
                        artifact_link_for_example(base_links, example),
                        "Base Artifact",
                    ),
                    artifact_markdown(
                        artifact_link_for_example(pr_links, example),
                        "PR Artifact",
                    ),
                )
            )
    else:
        lines.extend(
            [
                (
                    "The following files contain the validation output for the "
                    "checked Example benchmark YAML files."
                ),
                "",
            ]
        )
        for source_file in report.source_files:
            lines.append("- `{}`".format(source_file))
    lines.extend(["", "</details>"])
    return rendered.rstrip() + "\n" + "\n".join(lines).rstrip() + "\n"


def example_folder_markdown(
    example: ExampleResult,
    inventory_example: Optional[dict],
) -> str:
    path = str(
        (inventory_example or {}).get("path") or example.path
    ).rstrip("/")
    label = str(
        (inventory_example or {}).get("example") or example_category(path)
    )
    return source_path_markdown(label, path, "tree")


def benchmark_file_markdown(
    example: ExampleResult,
    inventory_example: Optional[dict],
) -> str:
    benchmark_file = str(
        (inventory_example or {}).get("benchmark_file") or ""
    ).rstrip("/")
    if not benchmark_file:
        return "`{}`".format(escape_table(example.name))
    return source_path_markdown(example.name, benchmark_file, "blob")


def source_path_markdown(label: str, path: str, kind: str) -> str:
    server_url = os.environ.get("GITHUB_SERVER_URL", "").rstrip("/")
    repository = os.environ.get("GITHUB_REPOSITORY", "").strip("/")
    source_sha = os.environ.get("GITHUB_SHA", "").strip()
    if server_url and repository and source_sha:
        target = "{}/{}/{}/{}/{}".format(
            server_url,
            repository,
            kind,
            quote(source_sha, safe=""),
            quote(path, safe="/"),
        )
    else:
        target = path
    return "[{}]({})".format(escape_table(label), target)


def example_category(path: str) -> str:
    parts = [part for part in path.strip("/").split("/") if part]
    if len(parts) > 1 and parts[0] == "examples":
        return parts[1]
    return parts[0] if parts else path


def example_artifact_key(name: str, path: str) -> str:
    return "{}\0{}".format(name, path.rstrip("/"))


def artifact_link_for_example(
    artifact_links: Dict[str, str],
    example: ExampleResult,
) -> Optional[str]:
    return artifact_links.get(
        example_artifact_key(example.name, example.path)
    ) or artifact_links.get(example.path)


def artifact_markdown(url: Optional[str], label: str) -> str:
    if not url:
        return "—"
    return "[{}]({})".format(label, url)


def regression_examples(comparisons: Sequence[object]) -> List[str]:
    examples = []
    seen = set()
    for comparison in comparisons:
        if not isinstance(comparison, dict):
            continue
        example = str(comparison.get("example") or "")
        if not example or example in seen:
            continue
        seen.add(example)
        examples.append(example)
    return sorted(examples)


def count_regression_field(
    comparisons: Sequence[object],
    field: str,
    fallback_classification: str = "",
    example: str = "",
) -> int:
    count = 0
    for comparison in comparisons:
        if not isinstance(comparison, dict):
            continue
        if example and comparison.get("example") != example:
            continue
        if field in comparison:
            count += int(comparison.get(field) or 0)
            continue
        if fallback_classification and comparison.get("classification") == fallback_classification:
            count += int(comparison.get("issue_count") or 0)
    return count


def append_step_summary(rendered: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    with open(summary_path, "a", encoding="utf-8") as summary:
        summary.write(rendered)
        summary.write("\n")


def emit_annotations(
    report: CombinedReport,
    mode: str = MODE_DYNAMIC,
) -> None:
    for example in report.examples:
        for check in example.checks:
            is_warning = mode == MODE_STATIC and check.status == WARNING
            if check.status not in BLOCKING_STATUSES and not is_warning:
                continue
            file_name = check.file or infer_file_from_details(check.details) or example.path
            message = check.message or check.name
            title = "{}: {}".format(example.path, check.name)
            command = "warning" if is_warning else "error"
            sys.stdout.write(
                "::{command} file={file},title={title}::{message}\n".format(
                    command=command,
                    file=escape_command_property(file_name),
                    title=escape_command_property(title),
                    message=escape_command_value(message),
                )
            )


def infer_file_from_details(details: Sequence[str]) -> str:
    if not details:
        return ""
    first = details[0]
    if " -> " in first:
        return first.split(" -> ", 1)[0]
    if ":" in first:
        return first.split(":", 1)[0]
    return first


def maybe_update_pr_comment(rendered: str) -> None:
    context = github_context()
    if not context:
        LOGGER.info(
            "Not a pull_request event or GitHub context is incomplete; "
            "skipping PR comment."
        )
        return

    owner_repo, pr_number, token, api_url = context
    body = truncate_comment(rendered)
    comments_url = "{}/repos/{}/issues/{}/comments".format(api_url, owner_repo, pr_number)

    try:
        comments = github_request("GET", comments_url + "?per_page=100", token)
        existing_url = find_existing_comment_url(comments)
        if existing_url:
            github_request("PATCH", existing_url, token, {"body": body})
            LOGGER.info("Updated Ianvs validation report comment on PR #%s.", pr_number)
        else:
            github_request("POST", comments_url, token, {"body": body})
            LOGGER.info("Created Ianvs validation report comment on PR #%s.", pr_number)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
        LOGGER.error("Failed to update PR comment: %s", exc)


def github_context() -> Optional[Tuple[str, int, str, str]]:
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    if event_name not in GITHUB_PULL_REQUEST_EVENTS:
        return None

    token = os.environ.get("GITHUB_TOKEN", "")
    owner_repo = os.environ.get("GITHUB_REPOSITORY", "")
    event_path = os.environ.get("GITHUB_EVENT_PATH", "")
    api_url = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
    if not token or not owner_repo or not event_path:
        return None

    payload = json.loads(Path(event_path).read_text(encoding="utf-8"))
    pull_request = payload.get("pull_request") or {}
    pr_number = pull_request.get("number") or payload.get("number")
    if not pr_number:
        return None

    return owner_repo, int(pr_number), token, api_url


def github_request(
    method: str,
    url: str,
    token: str,
    payload: Optional[Dict[str, object]] = None,
):
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": "Bearer {}".format(token),
            "Content-Type": "application/json",
            "User-Agent": GITHUB_USER_AGENT,
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        text = response.read().decode("utf-8")
        if not text:
            return None
        return json.loads(text)


def find_existing_comment_url(comments) -> str:
    if not isinstance(comments, list):
        return ""
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        body = comment.get("body", "")
        if COMMENT_MARKER in body:
            return str(comment.get("url", ""))
    return ""


def truncate_comment(rendered: str) -> str:
    if len(rendered) <= MAX_COMMENT_BODY_CHARS:
        return rendered
    suffix = "\n\n_Report truncated because it exceeded the GitHub comment size limit._\n"
    return rendered[: MAX_COMMENT_BODY_CHARS - len(suffix)] + suffix


def escape_table(value: str) -> str:
    return value.replace("|", "\\|")


def escape_command_value(value: str) -> str:
    return value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def escape_command_property(value: str) -> str:
    return (
        escape_command_value(value)
        .replace(":", "%3A")
        .replace(",", "%2C")
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] - %(message)s")
    raise SystemExit(main())
