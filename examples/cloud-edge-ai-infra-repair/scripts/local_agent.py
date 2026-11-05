# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Start/status/stop only a recorded local vLLM process, with bounded GPU use."""

import argparse
import json
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infra_repair.io_utils import http_json, save_json


def owned_process(record):
    path = Path("/proc/%d/cmdline" % record["pid"])
    if not path.exists():
        return False
    args = path.read_bytes().split(b"\0")
    return b"vllm.entrypoints.openai.api_server" in args and record["model_dir"].encode() in args


def stop(record):
    if not owned_process(record):
        raise RuntimeError("Recorded process is absent or ownership cannot be verified")
    os.killpg(record["pid"], signal.SIGTERM)
    for _ in range(150):
        if not owned_process(record):
            return
        time.sleep(0.1)
    if owned_process(record):
        os.killpg(record["pid"], signal.SIGKILL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "status", "stop"])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model-dir")
    parser.add_argument("--served-model-name", default="Qwen2.5-0.5B-Instruct")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.20)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--output", default="workspace/infra-repair/local-agent")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--foreground", action="store_true", help="Keep the WSL command alive after readiness")
    args = parser.parse_args()
    if not 0 < args.gpu_memory_utilization <= 0.4:
        parser.error("GPU memory utilization must be within (0, 0.4]")
    if sys.platform != "linux":
        parser.error("Run inside Linux/WSL")
    output = Path(args.output).resolve()
    record_path = output / "process.json"
    if args.action != "start":
        record = json.loads(record_path.read_text())
        if args.action == "stop":
            stop(record)
            print("Termination requested for recorded vLLM process")
        else:
            data = {**record, "owned_process_alive": owned_process(record)}
            try:
                data["models"] = http_json(record["base_url"] + "/models", timeout=5)
                data["http_ready"] = True
            except (URLError, TimeoutError, ConnectionError):
                data["http_ready"] = False
            print(json.dumps(data, indent=2))
        return
    if not args.model_dir or not (Path(args.model_dir) / "config.json").is_file():
        parser.error("Provide a complete local model snapshot with --model-dir")
    if record_path.exists() and owned_process(json.loads(record_path.read_text())):
        parser.error("Recorded agent is already running; use status")
    with socket.socket() as probe:
        probe.bind(("127.0.1.1", args.port))
    output.mkdir(parents=True, exist_ok=True)
    source = str(Path(args.model_dir).resolve())
    command = [args.python, "-m", "vllm.entrypoints.openai.api_server", "--model", source,
               "--served-model-name", args.served_model_name, "--host", "127.0.1.1",
               "--port", str(args.port), "--dtype", "half", "--max-model-len", "4096",
               "--max-num-seqs", "1", "--max-num-batched-tokens", "256",
               "--gpu-memory-utilization", str(args.gpu_memory_utilization), "--enforce-eager"]
    env = dict(os.environ)
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1"})
    # MRV2 uses UVA buffers, unavailable in some WSL GPU configurations.
    # Use the supported V1 runner switch, not a patch to the installed package.
    if "microsoft" in platform.release().lower():
        env["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    if (output / "server.log").exists():
        (output / "server.log").rename(output / ("server-%d.log" % time.time_ns()))
    with (output / "server.log").open("w") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, start_new_session=True, env=env)
    record = {"pid": process.pid, "model_dir": source, "model": args.served_model_name,
              "base_url": "http://127.0.1.1:%d/v1" % args.port, "command": command,
              "model_runner_v2": env.get("VLLM_USE_V2_MODEL_RUNNER", "default"),
              "flashinfer_sampler": env.get("VLLM_USE_FLASHINFER_SAMPLER", "default")}
    save_json(record_path, record)
    deadline = time.monotonic() + args.timeout
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("vLLM exited; inspect " + str(output / "server.log"))
            try:
                models = http_json(record["base_url"] + "/models", timeout=2)
                if record["model"] in [entry["id"] for entry in models["data"]]:
                    save_json(output / "ready.json", {"models": models, "base_url": record["base_url"]})
                    print(json.dumps(record, indent=2), flush=True)
                    if args.foreground:
                        process.wait()
                    return
            except (URLError, TimeoutError, ConnectionError):
                pass
            time.sleep(1)
        raise TimeoutError("vLLM startup deadline exceeded")
    except BaseException:
        if process.poll() is None:
            stop(record)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        raise


if __name__ == "__main__":
    main()
