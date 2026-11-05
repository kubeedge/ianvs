# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Flat and structured reports, with an optional existing Ianvs Rank adapter."""

import csv
from pathlib import Path
from types import SimpleNamespace

from .io_utils import save_json


def summarize(records):
    """Include failed attempts in every denominator; unavailable is None."""
    if not records:
        raise ValueError("Cannot rank an empty run")
    count = len(records)
    successful = [row["repair_seconds"] for row in records if row["repair_success"]]
    rollbacks = [row["rollback_success"] for row in records if row["rollback_success"] is not None]
    scenarios = sorted({row["scenario"] for row in records})
    per_scenario = {name: summarize_group([row for row in records if row["scenario"] == name]) for name in scenarios}
    return {"runs": count, "acceptance_eligible": False,
            "scope": "Batch summary; use acceptance with a separate complete rollback cohort",
            "repair_rate": sum(group["repair_rate"] for group in per_scenario.values()) / len(scenarios),
            "root_cause_accuracy": sum(group["root_cause_accuracy"] for group in per_scenario.values()) / len(scenarios),
            "injection_rate": sum(row["injection_success"] for row in records) / count,
            "cleanup_rate": sum(row["cleanup_success"] for row in records) / count,
            "rollback_rate": sum(rollbacks) / len(rollbacks) if rollbacks else None,
            "rollback_exercises": len(rollbacks),
            "mean_success_repair_seconds": sum(successful) / len(successful) if successful else None,
            "tool_calls": sum(row.get("tool_calls", 0) for row in records),
            "denied_calls": sum(row.get("denied_calls", 0) for row in records),
            "unauthorized_executions": sum(row.get("unauthorized_executions", 0) for row in records),
            "per_scenario": per_scenario}


def summarize_group(records):
    return {"runs": len(records), "repair_rate": sum(row["repair_success"] for row in records) / len(records),
            "root_cause_accuracy": sum(row["root_cause_correct"] for row in records) / len(records)}


def export(records, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize(records)
    save_json(output / "summary.json", summary)
    save_json(output / "results.json", records)
    rows = []
    for record in records:
        verification = record.get("verification") or {}
        rows.append({**{key: record.get(key) for key in (
            "run_id", "scenario", "agent", "status", "acceptance_eligible", "injection_success",
            "repair_success", "root_cause_correct", "repair_seconds", "attempt_seconds",
            "rollback_success", "cleanup_success", "tool_calls", "denied_calls", "failed_calls",
            "unauthorized_executions", "diagnosis_seconds", "tool_seconds", "approval_seconds",
            "verification_seconds", "repeated_calls")},
            **{key: verification.get(key) for key in ("slo_pass", "p95_latency_seconds", "p95_ttft_seconds",
                                                       "request_success_rate", "throughput_tokens_per_second")},
            "agent_tokens": record.get("agent_tokens", (record.get("diagnosis") or {}).get("tokens"))})
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def export_ianvs_rank(records, output):
    """Reuse Rank without importing learning paradigms or pretending CLI integration."""
    from core.storymanager.rank import Rank
    cases, results = [], {}
    for row in records:
        metrics = {"repair_success": int(row["repair_success"]),
                   "root_cause_correct": int(row["root_cause_correct"]),
                   "repair_seconds": row.get("repair_seconds"),
                   "tool_calls": row.get("tool_calls"),
                   "unauthorized_executions": row.get("unauthorized_executions"),
                   "slo_pass": (int(row["verification"]["slo_pass"]) if row.get("verification") else None)}
        algorithm = SimpleNamespace(name=row["agent"] + ":" + row["scenario"],
                                    paradigm_type="infrarepair-pilot", modules={})
        cases.append(SimpleNamespace(id=row["run_id"], algorithm=algorithm, output_dir=row["output_dir"]))
        results[row["run_id"]] = (metrics, row["started_at"])
    rank = Rank({"sort_by": [{"repair_success": "descend"}, {"root_cause_correct": "descend"}],
                 "selected_dataitem": {"paradigms": ["all"], "modules": ["all"],
                                       "hyperparameters": ["all"], "metrics": ["all"]},
                 "save_mode": "selected_and_all"})
    rank.save(cases, results, str(output))
    rank.plot()
