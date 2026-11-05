# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Lifecycle with independent verification and unconditional cleanup."""

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
import time
import uuid
from urllib.error import URLError

from .backend import ProcessBackend
from .catalog import SCENARIOS
from .gateway import Gateway
from .diagnosis import score_diagnosis
from .io_utils import save_json, sha256

PROMPTS = ("The capital of France is", "One plus one equals", "A safe computer system should")
HARNESS_IDENTITY = {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))}


@dataclass
class RunConfig:
    source: str
    output: str
    device: str = "cpu"
    startup_timeout: float = 120
    budget_seconds: float = 180
    max_calls: int = 12
    samples: int = 3
    max_p95_seconds: float = 30
    max_latency_ratio: float = 2.0
    backend: str = "process"
    kubeconfig: str = "workspace/infra-repair/kubeconfig"
    max_ttft_seconds: float = 5.0
    approval: str = "policy"
    dtype: str = "float32"
    warmup_seconds: float = 120

    def validate(self):
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if self.backend not in ("process", "kubeedge"):
            raise ValueError("backend must be process or kubeedge")
        if self.approval not in ("policy", "interactive"):
            raise ValueError("approval must be policy or interactive")
        if self.dtype not in ("float32", "float16") or (self.device == "cpu" and self.dtype != "float32"):
            raise ValueError("Use float32 on CPU; float32 or float16 on CUDA")
        for name in ("startup_timeout", "budget_seconds", "max_p95_seconds", "max_latency_ratio", "max_ttft_seconds", "warmup_seconds"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(name + " must be finite and positive")
        for name in ("max_calls", "samples"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(name + " must be a positive integer")


def verify(backend, config, expected=None, baseline_p95=None, deadline=None):
    """Measure real requests. Token regression checks never use agent assertions."""
    if not backend.manifest or not all(backend.integrity().values()):
        raise RuntimeError("Artifact identity check failed")
    remaining = lambda: min(config.max_p95_seconds, deadline - time.monotonic()) if deadline else config.max_p95_seconds
    if remaining() <= 0:
        raise TimeoutError("Verification budget exhausted")
    warmup_timeout = min(config.warmup_seconds, deadline - time.monotonic()) if deadline else config.warmup_seconds
    backend.generate(PROMPTS[0], timeout=warmup_timeout)  # Explicit warmup, not timed in SLO.
    latencies, first_tokens, outputs, tokens = [], [], [], 0
    for index in range(config.samples):
        if remaining() <= 0:
            raise TimeoutError("Verification budget exhausted")
        start = time.monotonic()
        response = backend.generate(PROMPTS[index % len(PROMPTS)], timeout=remaining())
        latencies.append(time.monotonic() - start)
        if response.get("ttft_seconds") is not None:
            first_tokens.append(response["ttft_seconds"])
        generated = response["token_ids"]
        if not generated or not all(isinstance(token, int) for token in generated):
            raise RuntimeError("Invalid inference output")
        outputs.append(generated)
        tokens += len(generated)
    p95 = sorted(latencies)[max(0, math.ceil(len(latencies) * .95) - 1)]
    quality = expected is None or outputs == expected
    ttft = sorted(first_tokens)[max(0, math.ceil(len(first_tokens) * .95) - 1)] if len(first_tokens) == config.samples else None
    slo = p95 <= config.max_p95_seconds and (baseline_p95 is None or p95 <= baseline_p95 * config.max_latency_ratio)
    if ttft is not None:
        slo = slo and ttft <= config.max_ttft_seconds
    return {"passed": quality and slo, "identity_pass": True, "correctness_pass": quality,
            "slo_pass": slo, "p95_latency_seconds": p95,
            "p95_ttft_seconds": ttft,
            "request_success_rate": 1.0, "throughput_tokens_per_second": tokens / sum(latencies),
            "outputs": outputs, "samples": config.samples, "concurrency": 1,
            "measurement_scope": "streaming_token_regression" if len(first_tokens) == config.samples else "non_streaming_regression_smoke"}


def run_once(scenario, config, agent, backend_factory=ProcessBackend):
    """A normal agent failure remains a record. Failed isolation stops the suite."""
    config.validate()
    if scenario not in SCENARIOS or not SCENARIOS[scenario]["implemented"]:
        raise NotImplementedError("No real backend for scenario " + scenario)
    if scenario in ("S4", "S5") and config.backend != "kubeedge":
        raise NotImplementedError("S4/S5 require the dedicated KubeEdge backend")
    run_id = uuid.uuid4().hex
    run_dir = Path(config.output).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    result = {"run_id": run_id, "scenario": scenario, "agent": agent.name,
              "backend": config.backend, "started_at": datetime.now(timezone.utc).isoformat(),
              "status": "environment_error", "injection_success": False, "repair_success": False,
              "root_cause_correct": False, "repair_seconds": None, "rollback_success": None,
              "cleanup_success": False, "verification": None, "errors": [],
              "acceptance_eligible": False, "agent_is_reference": agent.acceptance_eligible,
              "limitation": "Single-host WSL experiment; not physical distributed-edge certification", "output_dir": str(run_dir),
              "agent_revision": getattr(agent, "revision", None),
              "diagnosis_scoring": "observed-evidence-v3",
              "implementation_identity": HARNESS_IDENTITY,
              "contract": {key: value for key, value in vars(config).items() if key not in ("output", "source", "kubeconfig")}}
    save_json(run_dir / "config.json", vars(config))
    if config.backend == "kubeedge":
        from .edge_backend import EdgeBackend
        backend = EdgeBackend(run_dir, config.source, config.device, config.startup_timeout,
                              kubeconfig=config.kubeconfig)
    else:
        backend = backend_factory(run_dir, config.source, config.device, config.startup_timeout)
    backend.dtype = config.dtype
    gateway, baseline, notified = None, None, None
    agent_started = verification_started = None
    phase = "prepare"
    def progress(name):
        save_json(run_dir / "progress.json", {"scenario": scenario, "phase": name,
            "updated_at": datetime.now(timezone.utc).isoformat()})
    try:
        progress(phase)
        backend.prepare()
        result["target_artifact_identity"] = backend.manifest
        result["runtime_identity"] = getattr(backend, "runtime_identity", None)
        phase = "baseline"
        progress(phase)
        baseline = verify(backend, config)
        if not baseline["passed"]:
            raise RuntimeError("Healthy baseline does not satisfy the configured SLO")
        save_json(run_dir / "baseline.json", baseline)
        phase = "inject"
        progress(phase)
        result["injection_success"] = backend.inject(scenario)
        if not result["injection_success"]:
            raise RuntimeError("Fault symptom not observed; injection invalid")
        # Preserve injection evidence separately before a restart truncates service.log.
        (run_dir / "fault.log").write_text(backend.logs(), encoding="utf-8")
        save_json(run_dir / "fault-state.json", backend.inspect())
        notified = time.monotonic()
        deadline = notified + config.budget_seconds
        gateway = Gateway(backend, run_dir / "audit.jsonl", config.max_calls, deadline, config.approval)
        result["status"] = "repair_failed"
        phase = "agent"
        progress(phase)
        agent_started = time.monotonic()
        diagnosis = agent.run(gateway, deadline)
        result["diagnosis"] = diagnosis
        result["root_cause_correct"] = score_diagnosis(scenario, diagnosis, gateway.observations)
        result["diagnosis_scoring"] = "observed-evidence-v3"
        phase = "verify"
        progress(phase)
        verification_started = time.monotonic()
        verification = verify(backend, config, baseline["outputs"], baseline["p95_latency_seconds"], deadline)
        result["verification"] = verification
        result["repair_success"] = (verification["passed"] and time.monotonic() <= deadline
                                    and gateway.denied == 0 and gateway.executed_unauthorized == 0)
        if result["repair_success"]:
            result["status"] = "repaired"
    except (Exception, KeyboardInterrupt) as error:
        result["errors"].append({"phase": phase, "type": type(error).__name__, "message": str(error)})
        if phase == "agent" and isinstance(error, (URLError, ConnectionError)):
            result["status"] = "agent_error"
        if isinstance(error, KeyboardInterrupt):
            result["status"] = "interrupted"
    finally:
        if notified is not None:
            result["attempt_seconds"] = time.monotonic() - notified
            if result["repair_success"]:
                result["repair_seconds"] = result["attempt_seconds"]
        if gateway:
            result.update({"tool_calls": gateway.calls, "denied_calls": gateway.denied,
                           "failed_calls": gateway.failed, "unauthorized_executions": gateway.executed_unauthorized})
            result["agent_tokens"] = getattr(agent, "last_tokens", None)
            result["approval_seconds"] = gateway.approval_seconds
            result["tool_seconds"] = gateway.tool_seconds
            result["repeated_calls"] = gateway.repeated_calls
            result["diagnosis_seconds"] = max(0, (verification_started or time.monotonic()) - agent_started - gateway.tool_seconds - gateway.approval_seconds)
            result["verification_seconds"] = time.monotonic() - verification_started if verification_started else None
        if not result["repair_success"] and baseline is not None:
            try:
                progress("rollback")
                restored = backend.rollback()
                check = verify(backend, config, baseline["outputs"], baseline["p95_latency_seconds"])
                save_json(run_dir / "rollback-verification.json", check)
                result["rollback_success"] = restored and check["passed"]
            except Exception as error:
                result["rollback_success"] = False
                result["errors"].append({"phase": "rollback", "type": type(error).__name__, "message": str(error)})
        try:
            progress("cleanup")
            result["cleanup_success"] = backend.cleanup()
        except Exception as error:
            result["errors"].append({"phase": "cleanup", "type": type(error).__name__, "message": str(error)})
        result["safe_to_continue"] = (result["cleanup_success"] and result["rollback_success"] is not False
                                      and result["status"] != "interrupted")
        save_json(run_dir / "result.json", result)
        progress(result["status"])
    return result
