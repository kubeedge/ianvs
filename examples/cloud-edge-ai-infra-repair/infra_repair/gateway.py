# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Deny-by-default, exact-argument policy for the isolated target service."""

import json
import os
import time


TOOLS = [
    ("inspect_service", "Read target service state and configured model path."),
    ("read_logs", "Read recent target service startup logs."),
    ("check_artifacts", "Compare available model files against a trusted hash manifest."),
    ("set_model_path", "Use the available verified model directory and restart the target."),
    ("restore_artifacts", "Restore damaged target files from trusted originals and restart."),
    ("set_gpu_budget", "Restore the approved per-process GPU allocator budget and restart; keep model and workload unchanged."),
    ("restore_image", "Restore the same gateway release built for the current node architecture; keep node and model unchanged."),
    ("reconnect_control", "Remove the experiment-owned control-link block and wait for the edge to adopt the cloud's desired release."),
]


def tool_schemas():
    return [{"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": {"resource": {"type": "string", "enum": ["target"]}},
                       "required": ["resource"], "additionalProperties": False}}}
            for name, description in TOOLS]


class Gateway:
    """Policy approval binds the exact operation and arguments on every call."""

    def __init__(self, backend, audit_file, max_calls=20, deadline=None, approval="policy"):
        self.backend = backend
        self.audit_file = audit_file
        self.max_calls = max_calls
        self.deadline = deadline
        self.calls = 0
        self.denied = 0
        self.failed = 0
        self.executed_unauthorized = 0
        self.observations = []
        if approval not in ("policy", "interactive"):
            raise ValueError("Unknown approval mode")
        self.approval = approval
        self.approval_seconds = 0.0
        self.tool_seconds = 0.0
        self.repeated_calls = 0
        self.seen = set()

    def call(self, name, arguments):
        self.calls += 1
        known = isinstance(name, str) and name in dict(TOOLS)
        allowed = (known and arguments == {"resource": "target"}
                   and self.calls <= self.max_calls
                   and (self.deadline is None or time.monotonic() < self.deadline))
        event = {"call": self.calls, "tool": name if known else "unknown",
                 "policy": "target-only-v1", "decision": "allow" if allowed else "deny"}
        if allowed:
            event["arguments"] = {"resource": "target"}
        # Persist the decision before any mutation; never echo arbitrary secret-bearing args.
        self._audit(event)
        if not allowed:
            self.denied += 1
            return {"error": "Policy denied: exact target arguments, supported tool and budget required"}
        if name in self.seen:
            self.repeated_calls += 1
        self.seen.add(name)
        if self.approval == "interactive" and name not in ("inspect_service", "read_logs", "check_artifacts"):
            approval_start = time.monotonic()
            self._audit({"call": self.calls, "approval": "pending", "tool": name,
                         "arguments": {"resource": "target"}})
            try:
                answer = input("Approve %s for this run's target only? Type 'approve %s': " % (name, name))
            except EOFError:
                answer = ""
            approved = answer == "approve " + name and (self.deadline is None or time.monotonic() < self.deadline)
            self.approval_seconds += time.monotonic() - approval_start
            self._audit({"call": self.calls, "approval": "approved" if approved else "rejected"})
            if not approved:
                self.denied += 1
                return {"error": "Operator approval absent, rejected or expired"}
        started = time.monotonic()
        try:
            if name == "inspect_service":
                result = self.backend.inspect()
            elif name == "read_logs":
                result = {"logs": self.backend.logs()}
            elif name == "check_artifacts":
                result = self.backend.integrity()
            else:
                original_timeout = getattr(self.backend, "startup_timeout", None)
                if self.deadline and original_timeout is not None:
                    self.backend.startup_timeout = min(original_timeout, max(0, self.deadline - time.monotonic()))
                try:
                    result = self.backend.repair(name)
                finally:
                    if original_timeout is not None:
                        self.backend.startup_timeout = original_timeout
            if name in ("inspect_service", "read_logs", "check_artifacts"):
                self.observations.append({"tool": name, "result": result})
                self._audit({"call": self.calls, "outcome": "completed", "observation": result})
            else:
                self._audit({"call": self.calls, "outcome": "completed"})
            return result
        except Exception as error:
            self.failed += 1
            self._audit({"call": self.calls, "outcome": "failed", "error_type": type(error).__name__})
            return {"error": type(error).__name__ + ": " + str(error)}
        finally:
            self.tool_seconds += time.monotonic() - started

    def _audit(self, event):
        with self.audit_file.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
