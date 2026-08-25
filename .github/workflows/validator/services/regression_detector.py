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

"""Compare base and pull request validation results.

The detector blocks only failures introduced by the pull request. Failures that
already appear in the base result are reported as baseline debt so maintainers
can see them without forcing unrelated contributors to repair them.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


PASS = "PASS"
ERROR = "ERROR"
FAIL = "FAIL"
WARNING = "WARNING"
SKIP = "SKIP"
BLOCKING_STATUSES = {ERROR, FAIL}
DYNAMIC_ELIGIBILITY_CHECK = "Dynamic validation eligibility"
LOGGER = logging.getLogger("ianvs.validator.regression")

CLASS_PASSED = "Passed"
CLASS_PR_REGRESSION = "Failed: PR regression"
CLASS_PRE_EXISTING = "Failed: Pre-existing failure"
CLASS_FIXED_BASELINE = "Fixed: Pre-existing failure resolved"
CLASS_KNOWN_OR_BASELINE = "Failed: Known issue or baseline debt"
CLASS_UNKNOWN = "Unknown"

COMMENT_MARKER = "<!-- ianvs-example-regression-report -->"
MAX_COMMENT_BODY_CHARS = 60000
GITHUB_PULL_REQUEST_EVENTS = ("pull_request", "pull_request_target")
GITHUB_API_VERSION = "2022-11-28"
GITHUB_USER_AGENT = "ianvs-example-validator"
RUNTIME_SMOKE_TEST_PREFIX = "Runtime smoke test"
DETAIL_LOCATION_RE = re.compile(
    r"^(?P<file>.+?)\s+->\s+\(Line\s+(?P<line>\d+)\):\s*(?P<message>.*)$"
)
DIFF_HUNK_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
)
PYTHON_EXCEPTION_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception):\s+.+$"
)


@dataclass
class CheckResult:
    example: str
    name: str
    status: str
    message: str = ""
    file: str = ""
    details: List[str] = field(default_factory=list)
    example_name: str = ""

    @property
    def target_identity(self) -> Tuple[str, str]:
        return self.example_name or self.example, self.example

    @property
    def identity(self) -> Tuple[str, str]:
        return self.example, self.name

    @property
    def unit_check_identity(self) -> Tuple[str, str, str]:
        return self.target_identity + (self.name,)

    @property
    def issue_count(self) -> int:
        if self.status not in BLOCKING_STATUSES:
            return 0
        if self.name.startswith(RUNTIME_SMOKE_TEST_PREFIX):
            return 1
        return len(self.details) if self.details else 1


@dataclass(frozen=True)
class ErrorIssue:
    identity: Tuple[str, str, str, str]
    file: str = ""
    detail: str = ""


@dataclass(frozen=True)
class DetailLocation:
    file: str
    line: int
    message: str


@dataclass
class Comparison:
    example: str
    check: str
    classification: str
    base_status: str
    head_status: str
    cause: str
    blocks_pr: bool
    issue_count: int = 0
    base_issue_count: int = 0
    head_issue_count: int = 0
    base_warning_count: int = 0
    head_warning_count: int = 0
    pre_existing_issue_count: int = 0
    new_issue_count: int = 0
    fixed_issue_count: int = 0
    pre_existing_warning_count: int = 0
    new_warning_count: int = 0
    fixed_warning_count: int = 0
    message: str = ""
    file: str = ""
    details: List[str] = field(default_factory=list)
    pre_existing_details: List[str] = field(default_factory=list)
    new_details: List[str] = field(default_factory=list)
    fixed_details: List[str] = field(default_factory=list)
    pre_existing_warning_details: List[str] = field(default_factory=list)
    new_warning_details: List[str] = field(default_factory=list)
    fixed_warning_details: List[str] = field(default_factory=list)
    example_name: str = ""


@dataclass
class ValidationUnit:
    name: str
    path: str

    @property
    def identity(self) -> Tuple[str, str]:
        return self.name, self.path


@dataclass
class ExampleChange:
    change: str
    name: str
    path: str
    validation: str
    classification: str
    blocks_pr: bool
    inventory_status: str = ""
    previous_validation_state: str = ""


@dataclass
class RegressionReport:
    comparisons: List[Comparison]
    base_files: List[str]
    head_files: List[str]
    example_changes: List[ExampleChange] = field(default_factory=list)

    @property
    def blocks_pr(self) -> bool:
        return any(comparison.blocks_pr for comparison in self.comparisons) or any(
            change.blocks_pr for change in self.example_changes
        )

    @property
    def added_example_count(self) -> int:
        return sum(change.change == "Added" for change in self.example_changes)

    @property
    def removed_example_count(self) -> int:
        return sum(change.change == "Removed" for change in self.example_changes)

    def count_classification(self, classification: str) -> int:
        return sum(
            comparison.issue_count
            for comparison in self.comparisons
            if comparison.classification == classification
        )

    @property
    def current_error_count(self) -> int:
        return sum(comparison.head_issue_count for comparison in self.comparisons)

    @property
    def current_warning_count(self) -> int:
        return sum(comparison.head_warning_count for comparison in self.comparisons)

    @property
    def pre_existing_warning_count(self) -> int:
        return sum(
            comparison.pre_existing_warning_count for comparison in self.comparisons
        )

    @property
    def new_warning_count(self) -> int:
        return sum(comparison.new_warning_count for comparison in self.comparisons)

    @property
    def fixed_warning_count(self) -> int:
        return sum(comparison.fixed_warning_count for comparison in self.comparisons)

    @property
    def pre_existing_error_count(self) -> int:
        return sum(comparison.pre_existing_issue_count for comparison in self.comparisons)

    @property
    def new_error_count(self) -> int:
        return sum(comparison.new_issue_count for comparison in self.comparisons)

    @property
    def fixed_error_count(self) -> int:
        return sum(comparison.fixed_issue_count for comparison in self.comparisons)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare base and PR validation results for regressions."
    )
    parser.add_argument(
        "--base-results",
        action="append",
        default=[],
        help="Base branch JSON result file, directory, or glob. Can be repeated.",
    )
    parser.add_argument(
        "--head-results",
        action="append",
        default=[],
        help="Pull request JSON result file, directory, or glob. Can be repeated.",
    )
    parser.add_argument(
        "--base-ref",
        default="",
        help="Git revision used for the base validation results.",
    )
    parser.add_argument(
        "--head-ref",
        default="",
        help="Git revision used for the PR validation results.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional Markdown regression report output path.",
    )
    parser.add_argument(
        "--json-output",
        default="",
        help="Optional machine-readable regression report output path.",
    )
    parser.add_argument(
        "--step-summary",
        action="store_true",
        help="Append the Markdown report to GITHUB_STEP_SUMMARY when available.",
    )
    parser.add_argument(
        "--annotations",
        action="store_true",
        help="Emit GitHub Actions error annotations for new PR regressions.",
    )
    parser.add_argument(
        "--pr-comment",
        action="store_true",
        help="Create or update a pull request comment when running in a PR event.",
    )
    parser.add_argument(
        "--allow-missing-base",
        action="store_true",
        help="Continue when no base result files are found.",
    )
    parser.add_argument(
        "--allow-missing-head",
        action="store_true",
        help="Continue when no head result files are found.",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Exit 0 even when PR regressions are detected.",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Skip Markdown output, step summary, and PR comment rendering.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    base_paths = discover_result_paths(args.base_results)
    head_paths = discover_result_paths(args.head_results)

    if not head_paths and not args.allow_missing_head:
        LOGGER.error("No PR/head validation result JSON files were found.")
        return 2
    if not base_paths and not args.allow_missing_base:
        LOGGER.error("No base validation result JSON files were found.")
        return 2

    report = compare_results(
        base_paths=base_paths,
        head_paths=head_paths,
        base_ref=args.base_ref,
        head_ref=args.head_ref,
    )
    rendered = "" if args.json_only else render_markdown(report)
    publish_report(rendered, report, args)

    if report.blocks_pr and not args.no_fail:
        return 1
    return 0


def discover_result_paths(inputs: Sequence[str]) -> List[Path]:
    paths = []
    for value in inputs:
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


def compare_results(
    base_paths: Sequence[Path],
    head_paths: Sequence[Path],
    base_ref: str = "",
    head_ref: str = "",
) -> RegressionReport:
    base_units, base_checks = load_results(base_paths)
    head_units, head_checks = load_results(head_paths)
    comparisons = []
    line_mapper = GitLineMapper(base_ref, head_ref) if base_ref and head_ref else None
    skipped_units = dynamic_skipped_units(base_checks) | dynamic_skipped_units(
        head_checks
    )

    base_unit_ids = set(base_units)
    head_unit_ids = set(head_units)
    added_unit_ids = head_unit_ids - base_unit_ids
    removed_unit_ids = base_unit_ids - head_unit_ids
    example_changes = [
        added_example_change(head_units[identity], head_checks)
        for identity in sorted(added_unit_ids)
    ] + [
        removed_example_change(base_units[identity], base_checks)
        for identity in sorted(removed_unit_ids)
    ]

    identities = sorted(
        identity
        for identity in set(base_checks) | set(head_checks)
        if identity[:2] not in removed_unit_ids
        and identity[:2] not in skipped_units
    )
    for identity in identities:
        base_check = base_checks.get(identity)
        head_check = head_checks.get(identity)
        comparison = compare_check(
            base_check=base_check,
            head_check=head_check,
            line_mapper=line_mapper,
        )
        if comparison:
            comparisons.append(comparison)

    return RegressionReport(
        comparisons=comparisons,
        base_files=[path.as_posix() for path in base_paths],
        head_files=[path.as_posix() for path in head_paths],
        example_changes=example_changes,
    )


def dynamic_skipped_examples(
    checks: Dict[Tuple[str, str], CheckResult],
) -> set:
    return {
        check.example
        for check in checks.values()
        if check.name == DYNAMIC_ELIGIBILITY_CHECK and check.status == SKIP
    }


def dynamic_skipped_units(
    checks: Dict[Tuple[str, str, str], CheckResult],
) -> set:
    return {
        check.target_identity
        for check in checks.values()
        if check.name == DYNAMIC_ELIGIBILITY_CHECK and check.status == SKIP
    }


def load_results(
    paths: Sequence[Path],
) -> Tuple[
    Dict[Tuple[str, str], ValidationUnit],
    Dict[Tuple[str, str, str], CheckResult],
]:
    units: Dict[Tuple[str, str], ValidationUnit] = {}
    checks: Dict[Tuple[str, str, str], CheckResult] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for raw_example in payload.get("examples", []):
            if not isinstance(raw_example, dict):
                continue
            example_path = str(raw_example.get("path") or raw_example.get("name") or "")
            example_name = str(raw_example.get("name") or example_path)
            unit = ValidationUnit(name=example_name, path=example_path)
            units[unit.identity] = unit
            for raw_check in raw_example.get("checks", []):
                if not isinstance(raw_check, dict):
                    continue
                check = parse_check(
                    raw_check,
                    example=example_path,
                    example_name=example_name,
                )
                checks[check.unit_check_identity] = check
    return units, checks


def load_checks(paths: Sequence[Path]) -> Dict[Tuple[str, str], CheckResult]:
    checks = load_results(paths)[1]
    return {(check.example, check.name): check for check in checks.values()}


def parse_check(
    raw_check: Dict[str, object],
    example: str,
    example_name: str = "",
) -> CheckResult:
    return CheckResult(
        example=example,
        name=str(raw_check.get("name", "")),
        status=str(raw_check.get("status", "")),
        example_name=example_name,
        message=str(raw_check.get("message", "")),
        file=str(raw_check.get("file", "")),
        details=string_list(raw_check.get("details", [])),
    )


def added_example_change(
    unit: ValidationUnit,
    checks: Dict[Tuple[str, str, str], CheckResult],
) -> ExampleChange:
    unit_checks = checks_for_unit(checks, unit.identity)
    inventory_status = dynamic_inventory_status(unit_checks)
    if any(check.status in BLOCKING_STATUSES for check in unit_checks):
        return ExampleChange(
            change="Added",
            name=unit.name,
            path=unit.path,
            validation="Failed",
            classification=CLASS_PR_REGRESSION,
            blocks_pr=True,
            inventory_status=inventory_status,
        )
    if any(
        check.name == DYNAMIC_ELIGIBILITY_CHECK and check.status == SKIP
        for check in unit_checks
    ):
        return ExampleChange(
            change="Added",
            name=unit.name,
            path=unit.path,
            validation="Skipped",
            classification="Skipped / inactive",
            blocks_pr=False,
            inventory_status=inventory_status,
        )
    return ExampleChange(
        change="Added",
        name=unit.name,
        path=unit.path,
        validation="Passed",
        classification=CLASS_PASSED,
        blocks_pr=False,
        inventory_status=inventory_status,
    )


def removed_example_change(
    unit: ValidationUnit,
    checks: Dict[Tuple[str, str, str], CheckResult],
) -> ExampleChange:
    unit_checks = checks_for_unit(checks, unit.identity)
    return ExampleChange(
        change="Removed",
        name=unit.name,
        path=unit.path,
        validation="Removed",
        classification="Removed",
        blocks_pr=False,
        inventory_status=dynamic_inventory_status(unit_checks),
        previous_validation_state=validation_state(unit_checks),
    )


def checks_for_unit(
    checks: Dict[Tuple[str, str, str], CheckResult],
    identity: Tuple[str, str],
) -> List[CheckResult]:
    return [
        check
        for check_identity, check in checks.items()
        if check_identity[:2] == identity
    ]


def validation_state(checks: Sequence[CheckResult]) -> str:
    if any(check.status in BLOCKING_STATUSES for check in checks):
        return "Failed"
    if any(
        check.name == DYNAMIC_ELIGIBILITY_CHECK and check.status == SKIP
        for check in checks
    ):
        return "Skipped"
    return "Passed"


def dynamic_inventory_status(checks: Sequence[CheckResult]) -> str:
    for check in checks:
        if check.name != DYNAMIC_ELIGIBILITY_CHECK:
            continue
        for detail in check.details:
            prefix = "inventory status:"
            if detail.lower().startswith(prefix):
                return detail[len(prefix):].strip()
    return ""


def string_list(value: object) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def compare_check(
    base_check: Optional[CheckResult],
    head_check: Optional[CheckResult],
    line_mapper: Optional["GitLineMapper"] = None,
) -> Optional[Comparison]:
    if head_check is None and base_check is None:
        return None

    example = (head_check or base_check).example
    example_name = (head_check or base_check).example_name or example
    name = (head_check or base_check).name
    base_status = base_check.status if base_check else "MISSING"
    head_status = head_check.status if head_check else "MISSING"
    base_issues = error_issue_map(base_check)
    head_issues = error_issue_map(head_check)
    base_warnings = warning_issue_map(base_check)
    head_warnings = warning_issue_map(head_check)
    base_issue_count = len(base_issues)
    head_issue_count = len(head_issues)
    base_warning_count = len(base_warnings)
    head_warning_count = len(head_warnings)

    pre_existing_keys, new_keys, fixed_keys = compare_issue_keys(
        base_issues,
        head_issues,
        line_mapper,
    )

    pre_existing_details = issue_details(head_issues, pre_existing_keys)
    new_details = issue_details(head_issues, new_keys)
    fixed_details = issue_details(base_issues, fixed_keys)

    pre_existing_issue_count = len(pre_existing_keys)
    new_issue_count = len(new_keys)
    fixed_issue_count = len(fixed_keys)

    (
        pre_existing_warning_keys,
        new_warning_keys,
        fixed_warning_keys,
    ) = compare_issue_keys(base_warnings, head_warnings, line_mapper)
    pre_existing_warning_count = len(pre_existing_warning_keys)
    new_warning_count = len(new_warning_keys)
    fixed_warning_count = len(fixed_warning_keys)
    pre_existing_warning_details = issue_details(
        head_warnings,
        pre_existing_warning_keys,
    )
    new_warning_details = issue_details(
        head_warnings,
        new_warning_keys,
    )
    fixed_warning_details = issue_details(
        base_warnings,
        fixed_warning_keys,
    )
    current_warning_details = pre_existing_warning_details + new_warning_details

    if new_issue_count > 0:
        classification = CLASS_PR_REGRESSION
        issue_count = new_issue_count
        details = new_details
    elif head_issue_count > 0:
        classification = CLASS_PRE_EXISTING
        issue_count = pre_existing_issue_count
        details = pre_existing_details
    elif fixed_issue_count > 0:
        classification = CLASS_FIXED_BASELINE
        issue_count = fixed_issue_count
        details = fixed_details
    elif current_warning_details:
        classification = CLASS_PASSED
        issue_count = 0
        details = current_warning_details
    elif fixed_warning_details:
        classification = CLASS_PASSED
        issue_count = 0
        details = fixed_warning_details
    elif base_issue_count == 0 and head_issue_count == 0:
        classification = CLASS_PASSED
        issue_count = 0
        details = []
    else:
        classification = CLASS_UNKNOWN
        issue_count = max(base_issue_count, head_issue_count)
        details = issue_details(head_issues or base_issues, sorted(head_keys or base_keys))

    representative_check = head_check if head_check and head_status != "MISSING" else base_check
    file_name = first_issue_file(head_issues, new_keys) or first_issue_file(head_issues, pre_existing_keys)
    if not file_name:
        file_name = first_issue_file(base_issues, fixed_keys)
    if not file_name:
        file_name = first_issue_file(head_warnings, new_warning_keys)
    if not file_name:
        file_name = first_issue_file(head_warnings, pre_existing_warning_keys)
    if not file_name:
        file_name = first_issue_file(base_warnings, fixed_warning_keys)
    if not file_name and representative_check:
        file_name = representative_check.file

    return Comparison(
        example=example,
        example_name=example_name,
        check=name,
        classification=classification,
        base_status=base_status,
        head_status=head_status,
        cause=classify_cause(representative_check) if representative_check else "Unknown",
        blocks_pr=new_issue_count > 0,
        issue_count=issue_count,
        base_issue_count=base_issue_count,
        head_issue_count=head_issue_count,
        base_warning_count=base_warning_count,
        head_warning_count=head_warning_count,
        pre_existing_issue_count=pre_existing_issue_count,
        new_issue_count=new_issue_count,
        fixed_issue_count=fixed_issue_count,
        pre_existing_warning_count=pre_existing_warning_count,
        new_warning_count=new_warning_count,
        fixed_warning_count=fixed_warning_count,
        message=representative_check.message if representative_check else "",
        file=file_name,
        details=details,
        pre_existing_details=pre_existing_details,
        new_details=new_details,
        fixed_details=fixed_details,
        pre_existing_warning_details=pre_existing_warning_details,
        new_warning_details=new_warning_details,
        fixed_warning_details=fixed_warning_details,
    )


class GitLineMapper:
    """Maps unchanged base-file lines to their locations at the PR revision."""

    def __init__(self, base_ref: str, head_ref: str) -> None:
        self.base_ref = base_ref
        self.head_ref = head_ref
        self.hunks_by_file: Dict[str, List[Tuple[int, int, int]]] = {}
        self.available = git_revisions_exist(base_ref, head_ref)

    def maps_to(self, base_detail: str, head_detail: str) -> bool:
        if not self.available:
            return False
        base_location = parse_detail_location(base_detail)
        head_location = parse_detail_location(head_detail)
        if not base_location or not head_location:
            return False
        if (
            base_location.file != head_location.file
            or base_location.message != head_location.message
        ):
            return False
        mapped_line = self.map_line(base_location.file, base_location.line)
        return mapped_line == head_location.line

    def map_line(self, file_path: str, base_line: int) -> Optional[int]:
        offset = 0
        for old_start, old_count, new_count in self.hunks(file_path):
            if base_line < old_start:
                return base_line + offset
            if old_count and old_start <= base_line < old_start + old_count:
                return None
            offset += new_count - old_count
        return base_line + offset

    def hunks(self, file_path: str) -> List[Tuple[int, int, int]]:
        if file_path not in self.hunks_by_file:
            self.hunks_by_file[file_path] = git_diff_hunks(
                self.base_ref,
                self.head_ref,
                file_path,
            )
        return self.hunks_by_file[file_path]


def compare_issue_keys(
    base_issues: Dict[Tuple[str, str, str, str], ErrorIssue],
    head_issues: Dict[Tuple[str, str, str, str], ErrorIssue],
    line_mapper: Optional[GitLineMapper],
) -> Tuple[
    List[Tuple[str, str, str, str]],
    List[Tuple[str, str, str, str]],
    List[Tuple[str, str, str, str]],
]:
    """Compare issue sets, treating unchanged Git-mapped lines as one issue."""
    base_keys = set(base_issues)
    head_keys = set(head_issues)
    matched_base_keys = set(base_keys & head_keys)
    matched_head_keys = set(matched_base_keys)
    unmatched_base_keys = sorted(base_keys - matched_head_keys)
    unmatched_head_keys = sorted(head_keys - matched_head_keys)

    if line_mapper:
        remaining_head_keys = set(unmatched_head_keys)
        for base_key in unmatched_base_keys:
            base_issue = base_issues[base_key]
            matching_head_key = next(
                (
                    head_key
                    for head_key in sorted(remaining_head_keys)
                    if line_mapper.maps_to(base_issue.detail, head_issues[head_key].detail)
                ),
                None,
            )
            if matching_head_key is not None:
                matched_base_keys.add(base_key)
                matched_head_keys.add(matching_head_key)
                remaining_head_keys.remove(matching_head_key)

    pre_existing_keys = sorted(matched_head_keys)
    new_keys = sorted(head_keys - matched_head_keys)
    fixed_keys = sorted(base_keys - matched_base_keys)
    return pre_existing_keys, new_keys, fixed_keys


def parse_detail_location(detail: str) -> Optional[DetailLocation]:
    match = DETAIL_LOCATION_RE.match(detail)
    if not match:
        return None
    return DetailLocation(
        file=match.group("file"),
        line=int(match.group("line")),
        message=match.group("message"),
    )


def git_revisions_exist(base_ref: str, head_ref: str) -> bool:
    for revision in (base_ref, head_ref):
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--verify", "{}^{{commit}}".format(revision)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            return False
        if result.returncode != 0:
            return False
    return True


def git_diff_hunks(
    base_ref: str,
    head_ref: str,
    file_path: str,
) -> List[Tuple[int, int, int]]:
    try:
        result = subprocess.run(
            [
                "git",
                "diff",
                "--no-ext-diff",
                "--unified=0",
                base_ref,
                head_ref,
                "--",
                file_path,
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return []
    if result.returncode not in (0, 1):
        return []
    hunks = []
    for line in result.stdout.splitlines():
        match = DIFF_HUNK_RE.match(line)
        if not match:
            continue
        hunks.append(
            (
                int(match.group("old_start")),
                int(match.group("old_count") or 1),
                int(match.group("new_count") or 1),
            )
        )
    return hunks


def error_issue_map(check: Optional[CheckResult]) -> Dict[Tuple[str, str, str, str], ErrorIssue]:
    if not check or check.status not in BLOCKING_STATUSES:
        return {}

    if check.name.startswith(RUNTIME_SMOKE_TEST_PREFIX):
        detail = runtime_smoke_error_detail(check)
        identity = (check.example, check.name, "", detail)
        return {
            identity: ErrorIssue(
                identity=identity,
                file=check.file,
                detail=detail,
            )
        }

    if check.details:
        return {
            (check.example, check.name, check.file, detail): ErrorIssue(
                identity=(check.example, check.name, check.file, detail),
                file=check.file,
                detail=detail,
            )
            for detail in check.details
        }

    if check.file:
        identity = (check.example, check.name, check.file, "")
        return {identity: ErrorIssue(identity=identity, file=check.file, detail=check.file)}

    detail = check.message or check.name
    identity = (check.example, check.name, "", detail)
    return {identity: ErrorIssue(identity=identity, detail=detail)}


def runtime_smoke_error_detail(check: CheckResult) -> str:
    for detail in check.details:
        normalized = detail.strip()
        if PYTHON_EXCEPTION_RE.match(normalized):
            return normalized
    return check.message or next(
        (detail.strip() for detail in reversed(check.details) if detail.strip()),
        check.name,
    )


def warning_issue_map(check: Optional[CheckResult]) -> Dict[Tuple[str, str, str, str], ErrorIssue]:
    if not check or check.status != WARNING:
        return {}

    if check.details:
        return {
            (check.example, check.name, check.file, detail): ErrorIssue(
                identity=(check.example, check.name, check.file, detail),
                file=check.file,
                detail=detail,
            )
            for detail in check.details
        }

    if check.file:
        identity = (check.example, check.name, check.file, "")
        return {identity: ErrorIssue(identity=identity, file=check.file, detail=check.file)}

    detail = check.message or check.name
    identity = (check.example, check.name, "", detail)
    return {identity: ErrorIssue(identity=identity, detail=detail)}


def issue_details(
    issues: Dict[Tuple[str, str, str, str], ErrorIssue],
    keys: Sequence[Tuple[str, str, str, str]],
) -> List[str]:
    return [issues[key].detail for key in keys if key in issues and issues[key].detail]


def first_issue_file(
    issues: Dict[Tuple[str, str, str, str], ErrorIssue],
    keys: Sequence[Tuple[str, str, str, str]],
) -> str:
    for key in keys:
        issue = issues.get(key)
        if not issue:
            continue
        if issue.file:
            return issue.file
        location = parse_detail_location(issue.detail)
        if location:
            return location.file
    return ""


def classify_cause(check: CheckResult) -> str:
    text = " ".join([check.name, check.message, check.file, " ".join(check.details)]).lower()
    if "cuda" in text or "gpu" in text or "hardware" in text:
        return "Failed: Hardware assumption"
    if "metric" in text or "empty-pair" in text:
        return "Failed: Metric edge case"
    if "model" in text:
        return "Failed: Model/resource drift"
    if "dataset" in text or "jsonl" in text:
        return "Failed: Dataset/resource drift"
    if "dependency" in text or "requirements" in text or "package" in text:
        return "Failed: Dependency drift"
    if "yaml" in text or "path" in text or "readme" in text:
        return "Failed: Known issue"
    return CLASS_KNOWN_OR_BASELINE


def render_markdown(report: RegressionReport) -> str:
    lines = [
        COMMENT_MARKER,
        "# Ianvs Example Regression Report",
        "",
        "## Summary",
        "",
        (
            "Compares the base branch and PR validation results to identify new failures. "
            "Pre-existing failures do not block validation."
        ),
        "",
        "**PR blocking:** {}".format("Yes" if report.blocks_pr else "No"),
        "",
        "- Added examples: {}".format(report.added_example_count),
        "- Removed examples: {}".format(report.removed_example_count),
        "",
        "| Current errors | Pre-existing errors | New errors | Fixed errors |",
        "|---:|---:|---:|---:|",
        "| {} | {} | {} | {} |".format(
            report.current_error_count,
            report.pre_existing_error_count,
            report.new_error_count,
            report.fixed_error_count,
        ),
        "",
    ]

    append_example_changes(lines, report.example_changes)

    blocking = [comparison for comparison in report.comparisons if comparison.new_issue_count > 0]
    baseline_debt = [
        comparison
        for comparison in report.comparisons
        if comparison.pre_existing_issue_count > 0 and comparison.new_issue_count == 0
    ]
    fixed = [
        comparison
        for comparison in report.comparisons
        if comparison.fixed_issue_count > 0 and comparison.head_issue_count == 0
    ]

    if blocking:
        lines.extend(["## PR Regressions", ""])
        for comparison in blocking:
            append_comparison(lines, comparison)

    if baseline_debt:
        lines.extend(["## Pre-existing Failures", ""])
        for comparison in baseline_debt:
            append_comparison(lines, comparison)

    if fixed:
        lines.extend(["## Fixed Baseline Failures", ""])
        for comparison in fixed:
            append_comparison(lines, comparison)

    if not blocking and not baseline_debt and not fixed:
        lines.append("No validation regressions or baseline failures were detected.")

    lines.extend(["", "## Local Reproduction", ""])
    examples = sorted({comparison.example for comparison in report.comparisons})
    for example in examples:
        if not example:
            continue
        lines.extend(
            [
                "```bash",
                "python .github/workflows/validator/validation_runner.py --static --example \"{}\" --format markdown".format(
                    example
                ),
                "```",
                "",
            ]
        )

    if report.base_files or report.head_files:
        lines.extend(["## Compared Result Files", ""])
        for source_file in report.base_files:
            lines.append("- Base: `{}`".format(source_file))
        for source_file in report.head_files:
            lines.append("- PR: `{}`".format(source_file))

    return "\n".join(lines).rstrip() + "\n"


def append_example_changes(
    lines: List[str],
    changes: Sequence[ExampleChange],
) -> None:
    if not changes:
        lines.extend(["Example changes: None", ""])
        return

    lines.extend(
        [
            "## Example Changes",
            "",
            "| Change | Example | Validation | Classification | Blocks PR |",
            "|---|---|---|---|---|",
        ]
    )
    for change in changes:
        result = change.validation
        if change.change == "Removed" and change.previous_validation_state:
            result = "Removed (base: {})".format(change.previous_validation_state)
        if change.inventory_status:
            result = "{} (`{}`)".format(result, change.inventory_status)
        lines.append(
            "| {} | `{}` (`{}`) | {} | {} | {} |".format(
                change.change,
                change.path,
                change.name,
                result,
                change.classification,
                "Yes" if change.blocks_pr else "No",
            )
        )
    lines.append("")


def append_comparison(lines: List[str], comparison: Comparison) -> None:
    lines.append(
        "### `{}` (`{}`)".format(comparison.example, comparison.example_name)
    )
    lines.append("")
    lines.append("- Check: `{}`".format(comparison.check))
    lines.append("- Classification: {}".format(comparison.classification))
    lines.append("- Cause: {}".format(comparison.cause))
    lines.append("- Base result: `{}`".format(comparison.base_status))
    lines.append("- PR result: `{}`".format(comparison.head_status))
    lines.append("- Base errors: {}".format(comparison.base_issue_count))
    lines.append("- PR errors: {}".format(comparison.head_issue_count))
    lines.append("- Pre-existing: {}".format(comparison.pre_existing_issue_count))
    lines.append("- New: {}".format(comparison.new_issue_count))
    lines.append("- Fixed: {}".format(comparison.fixed_issue_count))
    lines.append("- Counted errors: {}".format(comparison.issue_count))
    lines.append("- Blocks PR: {}".format("Yes" if comparison.blocks_pr else "No"))
    if comparison.file:
        lines.append("- File: `{}`".format(comparison.file))
    if comparison.message:
        lines.append("- Message: {}".format(comparison.message))
    append_detail_group(lines, "Pre-existing details", comparison.pre_existing_details)
    append_detail_group(lines, "New details", comparison.new_details)
    append_detail_group(lines, "Fixed details", comparison.fixed_details)
    lines.append("")


def append_detail_group(lines: List[str], title: str, details: Sequence[str]) -> None:
    if not details:
        return
    lines.append("")
    lines.append("#### {}".format(title))
    lines.append("")
    for detail in details[:10]:
        lines.append("- `{}`".format(detail))
    if len(details) > 10:
        lines.append("- ... {} more".format(len(details) - 10))


def render_json(report: RegressionReport) -> str:
    payload = {
        "blocks_pr": report.blocks_pr,
        "current_error_count": report.current_error_count,
        "current_warning_count": report.current_warning_count,
        "pre_existing_warning_count": report.pre_existing_warning_count,
        "new_warning_count": report.new_warning_count,
        "fixed_warning_count": report.fixed_warning_count,
        "pre_existing_error_count": report.pre_existing_error_count,
        "new_error_count": report.new_error_count,
        "fixed_error_count": report.fixed_error_count,
        "added_example_count": report.added_example_count,
        "removed_example_count": report.removed_example_count,
        "base_files": report.base_files,
        "head_files": report.head_files,
        "example_changes": [
            {
                "change": change.change,
                "name": change.name,
                "path": change.path,
                "validation": change.validation,
                "classification": change.classification,
                "blocks_pr": change.blocks_pr,
                "inventory_status": change.inventory_status,
                "previous_validation_state": change.previous_validation_state,
            }
            for change in report.example_changes
        ],
        "comparisons": [
            {
                "example": comparison.example,
                "example_name": comparison.example_name,
                "check": comparison.check,
                "classification": comparison.classification,
                "base_status": comparison.base_status,
                "head_status": comparison.head_status,
                "cause": comparison.cause,
                "blocks_pr": comparison.blocks_pr,
                "issue_count": comparison.issue_count,
                "base_issue_count": comparison.base_issue_count,
                "head_issue_count": comparison.head_issue_count,
                "base_warning_count": comparison.base_warning_count,
                "head_warning_count": comparison.head_warning_count,
                "pre_existing_issue_count": comparison.pre_existing_issue_count,
                "new_issue_count": comparison.new_issue_count,
                "fixed_issue_count": comparison.fixed_issue_count,
                "pre_existing_warning_count": comparison.pre_existing_warning_count,
                "new_warning_count": comparison.new_warning_count,
                "fixed_warning_count": comparison.fixed_warning_count,
                "message": comparison.message,
                "file": comparison.file,
                "details": comparison.details,
                "pre_existing_details": comparison.pre_existing_details,
                "new_details": comparison.new_details,
                "fixed_details": comparison.fixed_details,
                "pre_existing_warning_details": comparison.pre_existing_warning_details,
                "new_warning_details": comparison.new_warning_details,
                "fixed_warning_details": comparison.fixed_warning_details,
            }
            for comparison in report.comparisons
        ],
    }
    return json.dumps(payload, indent=2)


def publish_report(
    rendered: str,
    report: RegressionReport,
    args: argparse.Namespace,
) -> None:
    if not args.json_only:
        write_or_print_report(rendered, args.output)

    if args.json_output:
        write_json_report(report, args.json_output)

    if args.step_summary and not args.json_only:
        append_step_summary(rendered)

    if args.annotations:
        emit_annotations(report)

    if args.pr_comment and not args.json_only:
        maybe_update_pr_comment(rendered)


def write_or_print_report(rendered: str, output: str) -> None:
    if not output:
        sys.stdout.write(rendered)
        return

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    LOGGER.info("Regression report written to %s", output_path)


def write_json_report(report: RegressionReport, output: str) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_json(report), encoding="utf-8")
    LOGGER.info("Regression JSON written to %s", output_path)


def append_step_summary(rendered: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    with open(summary_path, "a", encoding="utf-8") as summary:
        summary.write(rendered)
        summary.write("\n")


def emit_annotations(report: RegressionReport) -> None:
    for comparison in report.comparisons:
        if comparison.new_issue_count <= 0:
            continue
        title = "{}: {}".format(comparison.classification, comparison.check)
        message = "{}; base={}, PR={}, new={}".format(
            comparison.cause,
            comparison.base_status,
            comparison.head_status,
            comparison.new_issue_count,
        )
        sys.stdout.write(
            "::{command} file={file},title={title}::{message}\n".format(
                command="error",
                file=escape_command_property(comparison.file or comparison.example),
                title=escape_command_property(title),
                message=escape_command_value(message),
            )
        )


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
            LOGGER.info("Updated Ianvs regression report comment on PR #%s.", pr_number)
        else:
            github_request("POST", comments_url, token, {"body": body})
            LOGGER.info("Created Ianvs regression report comment on PR #%s.", pr_number)
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
