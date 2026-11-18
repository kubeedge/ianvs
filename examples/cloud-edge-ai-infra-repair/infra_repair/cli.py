# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Command-line interface for the isolated repair pilot."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import uuid

from .agents import CalibrationController, FailureController, LocalAgent
from .triage_agent import TriageAgent
from .backend import recover
from .catalog import SCENARIOS
from .io_utils import save_json
from .report import export, export_ianvs_rank
from .runner import RunConfig, run_once


def preflight():
    """Only observe; do not install software or connect to an existing cluster."""
    results = {"platform": platform.platform(), "python": sys.version, "executable": sys.executable,
               "linux": sys.platform == "linux", "packages": {}, "commands": {}}
    for name in ("torch", "transformers", "safetensors", "vllm", "pandas", "PyYAML"):
        try:
            results["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            results["packages"][name] = None
    for name, args in {"docker": ["version", "--format", "{{.Server.Version}}"],
                       "nvidia-smi": ["--query-gpu=name,memory.total", "--format=csv,noheader"],
                       "kubectl": ["version", "--client", "-o", "json"]}.items():
        if not shutil.which(name):
            results["commands"][name] = {"status": "MISSING"}
            continue
        try:
            completed = subprocess.run([name, *args], capture_output=True, text=True, timeout=15, check=False)
            results["commands"][name] = {"status": "PASS" if completed.returncode == 0 else "ERROR",
                                          "output": completed.stdout[:4000], "error": completed.stderr[:1000]}
        except subprocess.TimeoutExpired:
            results["commands"][name] = {"status": "TIMEOUT"}
    return results


def main():
    parser = argparse.ArgumentParser(description="AI infrastructure repair pilot (not full acceptance)")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("preflight")
    check.add_argument("--output")
    commands.add_parser("catalog")
    recovery = commands.add_parser("recover")
    recovery.add_argument("run_dir")
    report = commands.add_parser("report")
    report.add_argument("batch_dir")
    report.add_argument("--ianvs-rank", action="store_true")
    acceptance = commands.add_parser("acceptance")
    acceptance.add_argument("--reference", required=True)
    acceptance.add_argument("--rollback", required=True)
    acceptance.add_argument("--output", required=True)
    run = commands.add_parser("run")
    run.add_argument("--config", help="Pilot YAML configuration; this is not an Ianvs job YAML")
    run.add_argument("--scenario", choices=list(SCENARIOS), nargs="+")
    run.add_argument("--model-dir", default=os.environ.get("IANVS_TARGET_MODEL_DIR"))
    run.add_argument("--output", default="workspace/infra-repair")
    run.add_argument("--repeat", type=int, default=3)
    run.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    run.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    run.add_argument("--backend", choices=["process", "kubeedge"], default="process")
    run.add_argument("--kubeconfig", default="workspace/infra-repair/kubeconfig")
    run.add_argument("--agent", choices=["local", "triage", "calibration", "failure"], default="local")
    run.add_argument("--agent-url", default="http://127.0.1.1:8001/v1")
    run.add_argument("--agent-model", default="Qwen2.5-0.5B-Instruct")
    run.add_argument("--agent-key-env")
    run.add_argument("--budget-seconds", type=float, default=180)
    run.add_argument("--startup-timeout", type=float, default=120)
    run.add_argument("--samples", type=int, default=3)
    run.add_argument("--max-calls", type=int, default=12)
    run.add_argument("--max-p95-seconds", type=float, default=30)
    run.add_argument("--max-latency-ratio", type=float, default=2)
    run.add_argument("--max-ttft-seconds", type=float, default=5)
    run.add_argument("--warmup-seconds", type=float, default=120)
    run.add_argument("--approval", choices=["policy", "interactive"], default="policy")
    run.add_argument("--ianvs-rank", action="store_true")
    args = parser.parse_args()
    if args.command == "acceptance":
        from .acceptance import assess_batches
        assessment = assess_batches(args.reference, args.rollback, args.output)
        print(json.dumps(assessment, indent=2))
        return 0 if assessment["passed"] else 1
    if args.command == "catalog":
        print(json.dumps(SCENARIOS, indent=2))
        return 0
    if args.command == "preflight":
        data = preflight()
        if args.output:
            save_json(args.output, data)
        print(json.dumps(data, indent=2))
        return 0 if data["linux"] else 2
    if args.command == "recover":
        if (Path(args.run_dir) / "edge-journal.json").exists():
            from .edge_backend import recover_edge
            recovered = recover_edge(args.run_dir)
        else:
            recovered = recover(args.run_dir)
        print(json.dumps(recovered, indent=2))
        return 0
    if args.command == "report":
        batch = Path(args.batch_dir)
        records = json.loads((batch / "results.json").read_text())
        print(json.dumps(export(records, batch), indent=2))
        if args.ianvs_rank:
            export_ianvs_rank(records, batch)
        return 0
    if args.config:
        import yaml
        with Path(args.config).open(encoding="utf-8") as stream:
            config_data = yaml.safe_load(stream)
        if not isinstance(config_data, dict) or set(config_data) != {"pilot"} or not isinstance(config_data["pilot"], dict):
            parser.error("Pilot YAML requires one 'pilot' mapping")
        allowed = {action.dest for action in run._actions} - {"help", "config"}
        for key, value in config_data["pilot"].items():
            if key not in allowed:
                parser.error("Unknown pilot setting: " + key)
            flag = "--" + key.replace("_", "-")
            if not any(item == flag or item.startswith(flag + "=") for item in sys.argv[1:]):
                setattr(args, key, value)
    if not isinstance(args.scenario, list) or not args.scenario or any(name not in SCENARIOS for name in args.scenario):
        parser.error("A nonempty list of supported scenarios is required")
    if args.agent not in ("local", "triage", "calibration", "failure"):
        parser.error("Unsupported agent")
    if not args.model_dir or not isinstance(args.repeat, int) or isinstance(args.repeat, bool) or args.repeat < 1:
        parser.error("--model-dir (or IANVS_TARGET_MODEL_DIR) and positive --repeat required")
    if any(not SCENARIOS[name]["implemented"] for name in args.scenario):
        parser.error("Selected scenario has no real backend implemented yet")
    batch = Path(args.output).resolve() / ("batch-" + uuid.uuid4().hex)
    config = RunConfig(args.model_dir, str(batch), args.device, args.startup_timeout,
                       args.budget_seconds, args.max_calls, args.samples,
                       args.max_p95_seconds, args.max_latency_ratio, args.backend, args.kubeconfig,
                       args.max_ttft_seconds, args.approval, args.dtype, args.warmup_seconds)
    config.validate()
    agent = (CalibrationController() if args.agent == "calibration" else FailureController()
             if args.agent == "failure" else (TriageAgent if args.agent == "triage" else LocalAgent)(
                 args.agent_url, args.agent_model, args.agent_key_env))
    if isinstance(agent, LocalAgent):
        save_json(batch / "agent-preflight.json", agent.preflight())
    records = []
    stop = False
    for scenario in args.scenario:
        for _ in range(args.repeat):
            record = run_once(scenario, config, agent)
            records.append(record)
            export(records, batch)  # Preserve partial suite results after every attempt.
            print(json.dumps({key: record[key] for key in ("run_id", "scenario", "status", "cleanup_success")}), flush=True)
            if not record["safe_to_continue"]:
                stop = True
                break
        if stop:
            break
    if args.ianvs_rank:
        export_ianvs_rank(records, batch)
    print("Reports: " + str(batch))
    expected = "repair_failed" if args.agent == "failure" else "repaired"
    return 0 if all(row["status"] == expected and row["safe_to_continue"] for row in records) else 1
