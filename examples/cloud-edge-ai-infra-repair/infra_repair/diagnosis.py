# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Evaluator-only scoring of cause, target and cited observable evidence."""

from .catalog import SCENARIOS


def score_diagnosis(scenario, diagnosis, observations):
    if diagnosis.get("cause") != SCENARIOS[scenario]["cause"] or diagnosis.get("object") != "target":
        return False
    cited = diagnosis.get("evidence")
    if not isinstance(cited, dict):
        return False
    tool, field, value = (cited.get(key) for key in ("tool", "field", "value"))
    verified = False
    state = {}
    for observation in observations:
        result = observation["result"]
        if observation["tool"] == "inspect_service":
            state.update(result)
        if observation["tool"] != tool:
            continue
        if tool == "read_logs" and field == "logs" and isinstance(value, str):
            verified = verified or bool(value and value in result.get("logs", ""))
        elif field in result:
            verified = verified or (type(result[field]) is type(value) and result[field] == value)
    if not verified:
        return False
    log = value.lower() if tool == "read_logs" and isinstance(value, str) else ""
    if scenario == "S1":
        return ("configured model directory does not exist" in log or
                tool == "inspect_service" and field == "configured_model" and
                state.get("configured_model") != state.get("available_model"))
    if scenario == "S2":
        return ("cuda" in log and "out of memory" in log or
                tool == "inspect_service" and field == "gpu_memory_fraction" and
                state.get("gpu_memory_fraction", 1) < state.get("approved_gpu_memory_fraction", 0))
    if scenario == "S3":
        return (tool == "check_artifacts" and value is False or
                any(token in log for token in ("safetensorerror", "invalid header", "header too", "incomplete metadata")))
    if scenario == "S4":
        return (any(token in log for token in ("exec format error", "no match for platform")) or
                tool == "inspect_service" and field == "image_architecture" and
                state.get("image_architecture") != state.get("node_architecture"))
    if scenario == "S5":
        return (tool == "inspect_service" and field in ("control_link_reachable", "edge_revision", "desired_revision")
                and state.get("control_link_reachable") is False and
                state.get("edge_revision") != state.get("desired_revision"))
    return False
