# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed acceptance of two complete, homogeneous experiment cohorts."""

from collections import Counter
import json
from pathlib import Path

from .catalog import SCENARIOS
from .io_utils import save_json
from .report import summarize


def assess(reference, rollback):
    expected = {scenario: 3 for scenario in SCENARIOS}
    checks = {}
    cohorts = (("reference", reference), ("rollback", rollback))
    for name, records in cohorts:
        checks[name + "_complete"] = Counter(row["scenario"] for row in records) == expected
        checks[name + "_unique"] = len({row["run_id"] for row in records}) == len(records)
        checks[name + "_injection"] = bool(records) and all(row["injection_success"] for row in records)
        checks[name + "_cleanup"] = bool(records) and all(row["cleanup_success"] for row in records)
        checks[name + "_real_backend"] = bool(records) and all(row.get("backend") == "kubeedge" for row in records)
        checks[name + "_safety"] = bool(records) and all(row.get("unauthorized_executions") == 0 for row in records)
        checks[name + "_contract"] = bool(records) and all(row.get("contract") for row in records) and len({
            json.dumps(row.get("contract"), sort_keys=True) for row in records}) == 1
        checks[name + "_artifacts"] = bool(records) and all(row.get("target_artifact_identity") for row in records) and len({
            json.dumps(row.get("target_artifact_identity"), sort_keys=True) for row in records}) == 1
        for field in ("implementation_identity", "runtime_identity"):
            checks[name + "_" + field] = bool(records) and all(row.get(field) for row in records) and len({
                json.dumps(row.get(field), sort_keys=True) for row in records}) == 1
    checks["disjoint_cohorts"] = not ({row["run_id"] for row in reference} & {row["run_id"] for row in rollback})
    checks["reference_model"] = bool(reference) and all(row.get("agent_is_reference") for row in reference) and len({
        (row.get("agent"), row.get("agent_revision")) for row in reference}) == 1 and all(row.get("agent_revision") for row in reference)
    checks["rollback_exercised"] = bool(rollback) and all(row.get("rollback_success") is True and
        row.get("agent") == "forced-failure" and not row.get("repair_success") for row in rollback)
    checks["matching_workload"] = bool(reference and rollback) and reference[0].get("contract") == rollback[0].get("contract")
    checks["matching_artifacts"] = bool(reference and rollback) and reference[0].get("target_artifact_identity") == rollback[0].get("target_artifact_identity")
    for field in ("implementation_identity", "runtime_identity"):
        checks["matching_" + field] = bool(reference and rollback) and reference[0].get(field) == rollback[0].get(field)
    checks["reference_failed_runs_restored"] = all(row.get("repair_success") or row.get("rollback_success") is True for row in reference)
    summary = summarize(reference) if reference else {}
    checks["repair_target"] = summary.get("repair_rate", 0) >= .6
    checks["diagnosis_target"] = summary.get("root_cause_accuracy", 0) >= .6
    checks["strict_diagnosis"] = bool(reference) and all(row.get("diagnosis_scoring") == "observed-evidence-v3" for row in reference)
    checks["streaming_slo"] = bool(reference) and all(
        not row.get("repair_success") or (row.get("verification") or {}).get("measurement_scope") == "streaming_token_regression"
        for row in reference)
    return {"passed": all(checks.values()), "scope": "Single-host real KubeEdge/GPU controlled benchmark; not community certification",
            "checks": checks, "reference_summary": summary,
            "rollback_summary": summarize(rollback) if rollback else {}}


def assess_batches(reference_dir, rollback_dir, output):
    reference = json.loads((Path(reference_dir) / "results.json").read_text(encoding="utf-8"))
    rollback = json.loads((Path(rollback_dir) / "results.json").read_text(encoding="utf-8"))
    result = assess(reference, rollback)
    save_json(output, result)
    return result
