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

"""Load example inventory and detect changed examples between git refs.

Usage:
    python .github/workflows/validator/services/inventory_loader.py \
        --base-ref upstream/main \
        --head-ref HEAD \
        --mode static \
        --inventory .github/workflows/validator/data/example_inventory.yaml

    python .github/workflows/validator/services/inventory_loader.py \
        --run-all \
        --inventory .github/workflows/validator/data/example_inventory.yaml

    python .github/workflows/validator/services/inventory_loader.py \
        --schedule \
        --mode dynamic \
        --inventory .github/workflows/validator/data/example_inventory.yaml

Modes:
    static:
        Select examples from the inventory when Python or YAML files under the
        example path changed. Static checks do not prepare datasets, so they
        include examples that are not yet active.

    dynamic:
        Select changed examples from the inventory, including examples that are
        not yet active so dynamic validation can report them as skipped. If
        files under core/ or .github/workflows/ changed, select all examples
        from the inventory.

GitHub Actions outputs:
    mode, run_all, examples_changed, base_examples_changed,
    head_examples_changed, base_changed_examples, head_changed_examples,
    base_validation_matrix, head_validation_matrix, changed_files
"""

import argparse
import json
import logging
import os
import re
import subprocess
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence


DEFAULT_INVENTORY_PATH = ".github/workflows/validator/data/example_inventory.yaml"
DEFAULT_PYTHON_VERSION = "3.8"
DYNAMIC_RUN_ALL_PREFIXES = ("core/", ".github/workflows/")
STATIC_TRACKED_FILE_SUFFIXES = (".py", ".yaml", ".yml")
MODE_STATIC = "static"
MODE_DYNAMIC = "dynamic"
TIER2_HEALTH_ARTIFACT = "tier2-base-health-results"
LOGGER = logging.getLogger("ianvs.validator.inventory")
HEALTH_RECORD_RE = re.compile(
    r"<!--\s*ianvs-example-health-record\s+(\{.*?\})\s*-->", re.DOTALL
)
SCHEDULE_ACTION_PUBLISH_TIER2 = "publish_tier2"
SCHEDULE_ACTION_RUN_TIER3 = "run_tier3"
SCHEDULE_ACTION_NONE = "none"
DEFAULT_TIER3_CADENCE_DAYS = 7


def git_lines(args: Sequence[str]) -> List[str]:
    return subprocess.check_output(["git", *args], text=True).splitlines()


def normalize_path(value: str) -> str:
    return value.strip().strip("\"'").rstrip("/")


def parse_scalar(value: str):
    value = value.strip()
    if value in ("true", "True"):
        return True
    if value in ("false", "False"):
        return False
    if value in ("null", "None", "~"):
        return None
    return value.strip("\"'")


def load_inventory_without_pyyaml(inventory_path: Path) -> Dict[str, List[dict]]:
    return load_inventory_text_without_pyyaml(
        inventory_path.read_text(encoding="utf-8")
    )


def load_inventory_text_without_pyyaml(text: str) -> Dict[str, List[dict]]:
    inventory = {"examples": []}
    current_example = None
    current_nested_key = None

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()

        if indent == 0 and line == "examples:":
            continue

        if indent == 0 and line.startswith("- "):
            current_example = {}
            current_nested_key = None
            inventory["examples"].append(current_example)
            line = line[2:]
            if line:
                key, value = line.split(":", 1)
                current_example[key.strip()] = parse_scalar(value)
            continue

        if current_example is None or ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if indent == 2 and value:
            current_example[key] = parse_scalar(value)
            current_nested_key = None
        elif indent == 2:
            current_example[key] = {}
            current_nested_key = key
        elif indent > 2 and current_nested_key:
            current_example[current_nested_key][key] = parse_scalar(value)

    return inventory


def load_inventory(inventory_path: Path) -> Dict[str, List[dict]]:
    return load_inventory_text(inventory_path.read_text(encoding="utf-8"))


def load_inventory_text(text: str) -> Dict[str, List[dict]]:
    try:
        import yaml
    except ImportError:
        return load_inventory_text_without_pyyaml(text)

    data = yaml.safe_load(text) or {}
    return data


def load_inventory_examples(inventory_path: Path, active_only: bool = True) -> List[dict]:
    return inventory_examples(load_inventory(inventory_path), active_only)


def load_inventory_examples_at_ref(
    inventory_path: Path,
    revision: str,
    active_only: bool = True,
) -> List[dict]:
    inventory_object = "{}:{}".format(revision, inventory_path.as_posix())
    object_check = subprocess.run(
        ["git", "cat-file", "-e", inventory_object],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if object_check.returncode != 0:
        # A validator rollout can add the inventory and its workflow in the
        # same PR. In that case the base revision is valid, but it has no
        # inventory yet, so the correct comparison target is an empty list.
        # Verify the revision separately so invalid refs and repository errors
        # are not silently treated as a missing inventory.
        subprocess.check_output(
            ["git", "rev-parse", "--verify", "{}^{{commit}}".format(revision)],
            text=True,
            stderr=subprocess.STDOUT,
        )
        return []

    inventory_text = subprocess.check_output(
        ["git", "show", inventory_object],
        text=True,
    )
    return inventory_examples(load_inventory_text(inventory_text), active_only)


def inventory_examples(
    inventory: Dict[str, List[dict]],
    active_only: bool = True,
) -> List[dict]:
    examples = inventory.get("examples") or []
    selected_examples = []

    for example in examples:
        if active_only and example.get("status", "active") != "active":
            continue

        path = example.get("path")
        if not path:
            continue

        selected_examples.append(normalize_example(example))

    return selected_examples


def normalize_example(example: dict) -> dict:
    normalized = dict(example)
    normalized["path"] = normalize_path(str(example["path"]))
    return normalized


def file_matches_path(file_name: str, target_path: str) -> bool:
    return file_name == target_path or file_name.startswith(target_path + "/")


def is_static_tracked_file(file_name: str) -> bool:
    normalized = normalize_path(file_name).lower()
    return normalized.endswith(STATIC_TRACKED_FILE_SUFFIXES)


def detect_static_examples(changed_files: Sequence[str], examples: Sequence[dict]) -> List[dict]:
    changed_examples = []

    for example in examples:
        example_path = example["path"]
        if any(file_matches_path(changed_file, example_path) for changed_file in changed_files):
            changed_examples.append(example)

    return changed_examples


def should_run_all_dynamic(changed_files: Sequence[str]) -> bool:
    return any(
        changed_file.startswith(DYNAMIC_RUN_ALL_PREFIXES) for changed_file in changed_files
    )


def example_selector(example: dict) -> str:
    return str(example.get("benchmark_file") or example.get("name") or example["path"])


def example_python_version(example: dict) -> str:
    """Return the benchmark Python version, falling back to the CI default."""
    configured_version = example.get("python_version")
    if configured_version is None or not str(configured_version).strip():
        return DEFAULT_PYTHON_VERSION
    return str(configured_version).strip()


def inventory_selection_report(
    mode: str,
    examples: Sequence[dict],
    run_all: bool,
    changed_files: Sequence[str] = (),
    base_ref: str = "",
    head_ref: str = "",
) -> dict:
    return {
        "mode": mode,
        "base_ref": base_ref,
        "head_ref": head_ref,
        "run_all": run_all,
        "changed_files": list(changed_files),
        "examples_changed": bool(examples),
        "changed_examples": [example_selector(example) for example in examples],
        "validation_matrix": [
            {
                "selector": example_selector(example),
                "python_version": example_python_version(example),
            }
            for example in examples
        ],
        "check_items": list(examples),
    }


def select_all_examples(mode: str, inventory_path: Path) -> dict:
    examples = load_inventory_examples(inventory_path)
    return inventory_selection_report(mode=mode, examples=examples, run_all=True)


def select_scheduled_examples(mode: str, inventory_path: Path) -> dict:
    examples = load_inventory_examples(inventory_path)
    return inventory_selection_report(mode=mode, examples=examples, run_all=True)


def parse_utc(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_utc(value: Optional[datetime]) -> str:
    if value is None:
        return ""
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def load_health_record(readme_path: Path) -> Optional[dict]:
    if not readme_path.is_file():
        return None
    match = HEALTH_RECORD_RE.search(readme_path.read_text(encoding="utf-8"))
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
        return {
            "validated_at": parse_utc(str(payload["validated_at"])),
            "source_run_id": int(payload.get("source_run_id", 0)),
            "source_tier": str(payload.get("source_tier", "")),
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(
            "Invalid example health record in {}: {}".format(readme_path, error)
        )


def load_health_metadata(metadata_path: Path) -> Optional[dict]:
    """Load validation timing from example-status/summary.json."""
    if not metadata_path.is_file():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        validated_at = parse_utc(str(payload["validated_at"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(
            "Invalid example status metadata in {}: {}".format(metadata_path, error)
        ) from error
    return {
        "source_run_id": 0,
        "source_sha": str(payload.get("commit", "")),
        "validated_at": validated_at,
    }


def github_json(url: str, token: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": "Bearer {}".format(token),
            "User-Agent": "ianvs-example-validator",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def load_tier2_health_artifacts(repository: str, token: str) -> List[dict]:
    query = urllib.parse.urlencode(
        {"name": TIER2_HEALTH_ARTIFACT, "per_page": "100"}
    )
    payload = github_json(
        "https://api.github.com/repos/{}/actions/artifacts?{}".format(
            repository, query
        ),
        token,
    )
    artifacts = payload.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise ValueError("GitHub artifact response has no artifacts list")
    return [artifact for artifact in artifacts if isinstance(artifact, dict)]


def select_pending_tier2_artifact(
    artifacts: Sequence[dict],
    health_record: Optional[dict],
) -> Optional[dict]:
    """Select the latest unpublished Tier 2 artifact.

    Looking back to the published health timestamp lets any run recover when publication itself had an infrastructure
    failure.
    """

    last_validated_at = health_record["validated_at"] if health_record else None
    published_run_id = health_record["source_run_id"] if health_record else 0
    candidates = []
    for artifact in artifacts:
        if artifact.get("expired"):
            continue
        try:
            created_at = parse_utc(str(artifact["created_at"]))
            artifact_id = int(artifact["id"])
            run_id = int((artifact.get("workflow_run") or {})["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if run_id == published_run_id:
            continue
        if last_validated_at and created_at <= last_validated_at:
            continue
        candidates.append(
            {
                "artifact_id": artifact_id,
                "run_id": run_id,
                "created_at": created_at,
            }
        )
    if not candidates:
        return None
    return max(candidates, key=lambda artifact: artifact["created_at"])


def scheduled_validation_plan(
    now: datetime,
    health_record: Optional[dict],
    tier2_artifact: Optional[dict],
    cadence_days: int = DEFAULT_TIER3_CADENCE_DAYS,
) -> dict:
    if tier2_artifact:
        created_at = tier2_artifact["created_at"]
        return {
            "action": SCHEDULE_ACTION_PUBLISH_TIER2,
            "reason": "An unpublished Tier 2 base result is available.",
            "tier2_artifact_id": tier2_artifact["artifact_id"],
            "tier2_run_id": tier2_artifact["run_id"],
            "tier2_created_at": format_utc(created_at),
            "last_validated_at": format_utc(
                health_record["validated_at"] if health_record else None
            ),
            "next_tier3_at": format_utc(created_at + timedelta(days=cadence_days)),
        }

    if health_record is None:
        return {
            "action": SCHEDULE_ACTION_RUN_TIER3,
            "reason": (
                "The ci-managed/example-health-status branch has no T2/T3 "
                "validation record."
            ),
            "tier2_artifact_id": 0,
            "tier2_run_id": 0,
            "tier2_created_at": "",
            "last_validated_at": "",
            "next_tier3_at": "",
        }

    validated_at = health_record["validated_at"]
    next_tier3_at = validated_at + timedelta(days=cadence_days)
    due = now.astimezone(timezone.utc) >= next_tier3_at
    return {
        "action": SCHEDULE_ACTION_RUN_TIER3 if due else SCHEDULE_ACTION_NONE,
        "reason": (
            "The example-status validation record is at least {} days old.".format(
                cadence_days
            )
            if due
            else "No unpublished Tier 2 result exists and Tier 3 is not due."
        ),
        "tier2_artifact_id": 0,
        "tier2_run_id": 0,
        "tier2_created_at": "",
        "last_validated_at": format_utc(validated_at),
        "next_tier3_at": format_utc(next_tier3_at),
    }


def export_schedule_outputs(plan: dict, output_path: str) -> None:
    with open(output_path, "a", encoding="utf-8") as output:
        for key in (
            "action",
            "reason",
            "tier2_artifact_id",
            "tier2_run_id",
            "tier2_created_at",
            "last_validated_at",
            "next_tier3_at",
        ):
            output.write("{}={}\n".format(key, plan.get(key, "")))


def detect_changes(
    base_ref: str,
    head_ref: str,
    mode: str,
    inventory_path: Path,
    base_inventory_ref: str = "",
) -> dict:
    head_examples = load_inventory_examples(
        inventory_path,
        active_only=False,
    )
    base_examples = (
        load_inventory_examples_at_ref(
            inventory_path,
            base_inventory_ref,
            active_only=False,
        )
        if base_inventory_ref
        else head_examples
    )
    changed_files = git_lines(["diff", "--name-only", base_ref, head_ref])
    if mode == MODE_STATIC:
        changed_files = [
            changed_file
            for changed_file in changed_files
            if is_static_tracked_file(changed_file)
        ]

    run_all = mode == MODE_DYNAMIC and should_run_all_dynamic(changed_files)
    base_changed_examples = detect_static_examples(changed_files, base_examples)
    head_changed_examples = detect_static_examples(changed_files, head_examples)
    base_selected_examples = base_examples if run_all else base_changed_examples
    head_selected_examples = head_examples if run_all else head_changed_examples
    base_report = inventory_selection_report(
        mode=mode,
        examples=base_selected_examples,
        run_all=run_all,
        changed_files=changed_files,
        base_ref=base_ref,
        head_ref=head_ref,
    )
    head_report = inventory_selection_report(
        mode=mode,
        examples=head_selected_examples,
        run_all=run_all,
        changed_files=changed_files,
        base_ref=base_ref,
        head_ref=head_ref,
    )
    return {
        **head_report,
        "examples_changed": bool(base_selected_examples or head_selected_examples),
        "base_examples_changed": bool(base_selected_examples),
        "head_examples_changed": bool(head_selected_examples),
        "base_changed_examples": base_report["changed_examples"],
        "head_changed_examples": head_report["changed_examples"],
        "base_validation_matrix": base_report["validation_matrix"],
        "head_validation_matrix": head_report["validation_matrix"],
        "base_check_items": base_report["check_items"],
        "head_check_items": head_report["check_items"],
    }


def export_github_outputs(report: dict, output_path: str) -> None:
    with open(output_path, "a", encoding="utf-8") as output:
        output.write("mode={}\n".format(report["mode"]))
        output.write(
            "run_all={}\n".format("true" if report["run_all"] else "false")
        )
        output.write(
            "examples_changed={}\n".format(
                "true" if report["examples_changed"] else "false"
            )
        )
        for key in ("base_examples_changed", "head_examples_changed"):
            if key in report:
                output.write(
                    "{}={}\n".format(key, "true" if report[key] else "false")
                )
        output.write(
            "changed_examples={}\n".format(json.dumps(report["changed_examples"]))
        )
        output.write(
            "validation_matrix={}\n".format(json.dumps(report["validation_matrix"]))
        )
        for key in (
            "base_changed_examples",
            "head_changed_examples",
            "base_validation_matrix",
            "head_validation_matrix",
            "base_check_items",
            "head_check_items",
        ):
            if key in report:
                output.write("{}={}\n".format(key, json.dumps(report[key])))
        output.write("changed_files={}\n".format(json.dumps(report["changed_files"])))
        output.write("check_items={}\n".format(json.dumps(report["check_items"])))


def print_report(report: dict) -> None:
    if report["base_ref"] and report["head_ref"]:
        LOGGER.info(
            "Detect mode: {mode}; comparing {base_ref}..{head_ref}".format(
                mode=report["mode"],
                base_ref=report["base_ref"],
                head_ref=report["head_ref"],
            )
        )
    else:
        LOGGER.info(
            "Detect mode: %s; using all inventory examples", report["mode"]
        )
    LOGGER.info("Run all inventory examples: %s", report["run_all"])

    if report["changed_files"]:
        LOGGER.info("Changed files:")
        for changed_file in report["changed_files"]:
            LOGGER.info("  - %s", changed_file)

    LOGGER.info("Examples to check:")
    for item in report["check_items"]:
        LOGGER.info(
            "  - %s: %s", item.get("name", item["path"]), item["path"]
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect inventory examples changed between two git refs."
    )
    parser.add_argument(
        "--base-ref",
        default="upstream/main",
        help="Base git ref to compare from.",
    )
    parser.add_argument(
        "--head-ref",
        default="HEAD",
        help="Head git ref to compare to.",
    )
    parser.add_argument(
        "--mode",
        choices=(MODE_STATIC, MODE_DYNAMIC),
        default=MODE_STATIC,
        help="Detection mode. Static checks example changes only. Dynamic also runs all inventory examples for core/workflow changes.",
    )
    parser.add_argument(
        "--inventory",
        default=DEFAULT_INVENTORY_PATH,
        help="Example inventory YAML path.",
    )
    parser.add_argument(
        "--base-inventory-ref",
        default="",
        help=(
            "Git revision from which to load the base inventory. When omitted, "
            "the current inventory is used for both target sets."
        ),
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Select all active inventory examples without detecting changed files.",
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Select active examples for scheduled dynamic validation.",
    )
    parser.add_argument(
        "--plan-schedule",
        action="store_true",
        help="Choose whether today's scheduled run publishes T2, runs T3, or waits.",
    )
    parser.add_argument("--readme", default="examples/README.md")
    parser.add_argument(
        "--health-metadata",
        default="",
        help="example-status summary.json; preferred over the legacy README record.",
    )
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""))
    parser.add_argument(
        "--cadence-days", type=int, default=DEFAULT_TIER3_CADENCE_DAYS
    )
    parser.add_argument("--now", default="", help="UTC ISO time override for tests.")
    parser.add_argument(
        "--github-output",
        default=os.environ.get("GITHUB_OUTPUT", ""),
        help="GitHub Actions output file. Defaults to GITHUB_OUTPUT.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    inventory_path = Path(args.inventory)
    if args.plan_schedule:
        if args.cadence_days <= 0:
            LOGGER.error("--cadence-days must be positive.")
            return 2
        if not args.repository or not args.token:
            LOGGER.error("--repository and --token are required for schedule planning.")
            return 2
        now = parse_utc(args.now) if args.now else datetime.now(timezone.utc)
        try:
            health_record = (
                load_health_metadata(Path(args.health_metadata))
                if args.health_metadata
                else load_health_record(Path(args.readme))
            )
            tier2_artifact = select_pending_tier2_artifact(
                load_tier2_health_artifacts(args.repository, args.token),
                health_record,
            )
            plan = scheduled_validation_plan(
                now, health_record, tier2_artifact, args.cadence_days
            )
        except (OSError, ValueError) as error:
            LOGGER.error("Unable to plan scheduled validation: %s", error)
            return 1
        LOGGER.info("Scheduled validation action: %s", plan["action"])
        LOGGER.info("Reason: %s", plan["reason"])
        if plan["next_tier3_at"]:
            LOGGER.info("Next Tier 3 time: %s", plan["next_tier3_at"])
        if args.github_output:
            export_schedule_outputs(plan, args.github_output)
        return 0
    if args.schedule:
        if args.mode != MODE_DYNAMIC:
            LOGGER.error("--schedule is only supported with --mode dynamic.")
            return 2
        report = select_scheduled_examples(mode=args.mode, inventory_path=inventory_path)
    elif args.run_all:
        report = select_all_examples(mode=args.mode, inventory_path=inventory_path)
    else:
        report = detect_changes(
            base_ref=args.base_ref,
            head_ref=args.head_ref,
            mode=args.mode,
            inventory_path=inventory_path,
            base_inventory_ref=args.base_inventory_ref,
        )

    print_report(report)
    if args.github_output:
        export_github_outputs(report, args.github_output)

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] - %(message)s")
    raise SystemExit(main())
