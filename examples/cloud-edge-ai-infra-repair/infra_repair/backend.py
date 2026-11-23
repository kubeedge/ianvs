# Copyright 2026 The KubeEdge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Isolated Linux process backend. Never touches an existing cluster or service."""

import json
import importlib.metadata
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import platform

from .io_utils import save_json, sha256, stream_tokens


class ProcessBackend:
    """Own exactly one copied model and one child process per run."""

    def __init__(self, run_dir, source, device="cpu", startup_timeout=120):
        self.run_dir = Path(run_dir).resolve()
        self.work = self.run_dir / "work"
        self.source = Path(source).resolve()
        self.model = self.work / "model"
        self.configured_model = self.model
        self.ready = self.work / "ready.json"
        self.log = self.run_dir / "service.log"
        self.journal = self.run_dir / "journal.json"
        self.device = device
        self.startup_timeout = startup_timeout
        self.process = None
        self.port = None
        self.manifest = {}
        self.gpu_memory_fraction = 0.4
        self.host = "127.0.1.1"
        self.dtype = "float32"
        self.runtime_identity = {"python": platform.python_version()}
        for name in ("torch", "transformers", "safetensors"):
            try:
                self.runtime_identity[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                self.runtime_identity[name] = None

    def prepare(self):
        if sys.platform != "linux":
            raise RuntimeError("Real process experiments require Linux/WSL")
        if not (self.source / "config.json").is_file():
            raise ValueError("Source must be a complete local model snapshot")
        if self.source == self.run_dir or self.run_dir in self.source.parents:
            raise ValueError("Model source must be outside the disposable run directory")
        if self.work.exists():
            raise ValueError("Refusing to reuse an existing run workspace")
        self.work.mkdir(parents=True)
        save_json(self.journal, {"state": "preparing", "pid": None, "ready": str(self.ready)})
        shutil.copytree(self.source, self.model, symlinks=False)
        self.manifest = {str(path.relative_to(self.model)): sha256(path)
                         for path in sorted(self.model.rglob("*")) if path.is_file()}
        save_json(self.run_dir / "artifact_manifest.json", self.manifest)
        self.start()

    def start(self):
        self.stop()
        self.ready.unlink(missing_ok=True)
        package_parent = str(Path(__file__).resolve().parents[1])
        env = dict(os.environ)
        env.update({"PYTHONPATH": package_parent, "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
        command = [sys.executable, "-m", "infra_repair.service", "--model",
                   str(self.configured_model), "--ready", str(self.ready),
                   "--device", self.device, "--gpu-memory-fraction", str(self.gpu_memory_fraction),
                   "--host", self.host, "--dtype", self.dtype]
        with self.log.open("w", encoding="utf-8") as stream:
            self.process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                            env=env, start_new_session=True)
        save_json(self.journal, {"state": "running", "pid": self.process.pid,
                                "ready": str(self.ready)})
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.port = None
                return False
            if self.ready.is_file():
                self.port = json.loads(self.ready.read_text())["port"]
                return True
            time.sleep(0.1)
        self.stop()
        raise TimeoutError("Model startup exceeded configured timeout")

    def stop(self):
        if self.process is not None and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10)
        self.port = None

    def inspect(self):
        return {"service": "target", "running": self.process is not None and self.process.poll() is None,
                "configured_model": str(self.configured_model), "available_model": str(self.model),
                "device": self.device, "dtype": self.dtype, "gpu_memory_fraction": self.gpu_memory_fraction,
                "approved_gpu_memory_fraction": 0.4}

    def logs(self):
        return self.log.read_text(encoding="utf-8", errors="replace")[-6000:] if self.log.exists() else ""

    def integrity(self):
        return {name: (self.model / name).is_file() and sha256(self.model / name) == digest
                for name, digest in self.manifest.items()}

    def generate(self, prompt, timeout=30):
        if self.port is None:
            raise RuntimeError("Inference service is unavailable")
        return stream_tokens("http://127.0.1.1:%d/generate-stream" % self.port,
                         {"prompt": prompt}, timeout=timeout)

    def inject(self, scenario):
        self.stop()
        if scenario == "S1":
            self.configured_model = self.work / "missing-model"
        elif scenario == "S2":
            if self.device != "cuda":
                raise ValueError("S2 requires a real CUDA device")
            self.gpu_memory_fraction = 0.02
        elif scenario == "S3":
            weights = sorted(self.model.glob("*.safetensors"))
            if not weights:
                raise ValueError("S3 requires safetensors weights")
            with weights[0].open("r+b") as stream:
                stream.truncate(32)
        else:
            raise NotImplementedError("Scenario has no real process backend: " + scenario)
        started = self.start()
        logs = self.logs().lower()
        if scenario == "S1":
            evidence = "configured model directory does not exist" in logs
        elif scenario == "S2":
            evidence = "out of memory" in logs and "cuda" in logs
        else:
            evidence = not all(self.integrity().values()) and any(
                token in logs for token in ("safetensor", "header", "deserialize"))
        return not started and evidence

    def repair(self, action):
        if action == "set_model_path":
            self.configured_model = self.model
        elif action == "set_gpu_budget":
            self.gpu_memory_fraction = 0.4
        elif action == "restore_artifacts":
            self.stop()
            for name, valid in self.integrity().items():
                if not valid:
                    source = self.source / name
                    if sha256(source) != self.manifest[name]:
                        raise RuntimeError("Trusted source changed since baseline")
                    target = self.model / name
                    temporary = target.with_suffix(target.suffix + ".restore")
                    shutil.copyfile(source, temporary)
                    temporary.replace(target)
        else:
            raise ValueError("Unknown repair action")
        return {"started": self.start()}

    def rollback(self):
        self.configured_model = self.model
        self.gpu_memory_fraction = 0.4
        self.repair("restore_artifacts")
        return all(self.integrity().values()) and self.port is not None

    def cleanup(self):
        self.stop()
        if self.work.is_symlink():
            raise RuntimeError("Refusing cleanup of symlink workspace")
        if self.work.exists():
            shutil.rmtree(self.work)
        clean = not self.work.exists() and (self.process is None or self.process.poll() is not None)
        save_json(self.journal, {"state": "cleaned" if clean else "cleanup_failed", "pid": None,
                                "ready": str(self.ready)})
        return clean


def recover(run_dir):
    """Conservatively clean an interrupted run, checking PID ownership first."""
    run_dir = Path(run_dir).resolve()
    journal = run_dir / "journal.json"
    state = json.loads(journal.read_text())
    work = run_dir / "work"
    expected_ready = str(work / "ready.json")
    if state.get("ready") != expected_ready or work.is_symlink():
        raise RuntimeError("Journal does not match this run directory")
    pid = state.get("pid")
    if pid and Path("/proc/%d" % pid).exists():
        command = Path("/proc/%d/cmdline" % pid).read_bytes().split(b"\0")
        if b"infra_repair.service" not in command or expected_ready.encode() not in command:
            raise RuntimeError("PID ownership cannot be verified; no process was killed")
        os.kill(pid, signal.SIGTERM)
        for _ in range(100):
            if not Path("/proc/%d" % pid).exists():
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("Owned service still exists; manual investigation required")
    if work.exists():
        shutil.rmtree(work)
    save_json(journal, {"state": "recovered_cleanup", "pid": None, "ready": expected_ready})
    return {"cleaned": True, "rollback_verified": False}
