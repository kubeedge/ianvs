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

"""System-level metrics for the simulation sandbox: process-tree memory/CPU
sampling and OOM detection.

Peak memory is reported as PSS (proportional set size) where the kernel
exposes it, and summed RSS otherwise, with the source recorded as
``memory_source`` in the output. Summing RSS across a process tree
double-counts shared pages (interpreter text, forked model weights); PSS
divides each shared page by the number of mappers, so the sum is meaningful.
"""

import re
import threading
import time

try:
    import psutil
except ImportError:  # pragma: no cover - exercised when psutil isn't installed
    psutil = None

_OOM_EXIT_CODES = {-9, 137}
_OOM_MARKERS = re.compile(r"out of memory|memoryerror|oomkilled", re.IGNORECASE)


def detect_oom_kill(returncode, stderr_text=""):
    """
    Whether a worker's exit looks like it was killed for memory.

    True for SIGKILL (``-9``) or its shell-reported form (``137``), or when
    ``stderr_text`` contains a recognisable OOM marker. False for a clean
    exit or an ordinary application failure.
    """
    if returncode in _OOM_EXIT_CODES:
        return True
    return bool(stderr_text) and _OOM_MARKERS.search(stderr_text) is not None


# pylint: disable=too-many-instance-attributes
class ProfileResult:
    """The system metrics collected for one sandboxed test case."""

    def __init__(self):
        self.peak_memory_bytes = None
        self.mean_memory_bytes = None
        self.quota_memory_bytes = None
        self.cpu_time_s = None
        self.cpu_utilization_pct = None
        self.wall_time_s = None
        self.memory_source = None
        self.sample_count = 0
        self.exit_code = None
        self.oom_killed = False
        self.succeeded = False
        self.error = None

    def memory_headroom_pct(self):
        """
        Percentage of the memory quota left unused; negative if exceeded.

        ``None`` when no quota was declared, since "headroom" is meaningless
        without a budget to measure it against. Negative values are
        reported as-is rather than clamped to zero, because clamping would
        hide the one finding an edge deployment engineer needs most.
        """
        if not self.quota_memory_bytes:
            return None
        peak = self.peak_memory_bytes or 0
        return round((1.0 - peak / self.quota_memory_bytes) * 100.0, 4)

    def as_dict(self):
        """A flat, JSON-safe dict, merged into the test case's own metrics."""

        def to_mb(value):
            return round(value / 2 ** 20, 2) if value is not None else None

        return {
            "peak_memory_mb": to_mb(self.peak_memory_bytes),
            "mean_memory_mb": to_mb(self.mean_memory_bytes),
            "quota_memory_mb": to_mb(self.quota_memory_bytes),
            "cpu_time_s": self.cpu_time_s,
            "cpu_utilization_pct": self.cpu_utilization_pct,
            "wall_time_s": self.wall_time_s,
            "memory_headroom_pct": self.memory_headroom_pct(),
            "memory_source": self.memory_source,
            "sample_count": self.sample_count,
            "exit_code": self.exit_code,
            "oom_killed": self.oom_killed,
        }


# pylint: disable=too-many-instance-attributes
class TreeSampler:
    """
    Background sampler for a process tree's memory and CPU.

    Runs in the parent process (not the worker being measured), so a worker
    that is OOM-killed mid-run still yields the samples taken up to the
    moment it died — exactly the case where the measurement matters most.
    Degrades to wall-clock-only (no samples) when ``psutil`` isn't
    installed, rather than failing the run.
    """

    def __init__(self, pid, interval=0.05):
        self.pid = pid
        self.interval = interval
        self._peak_bytes = 0
        self._samples = []
        self._cpu_times = []
        self._memory_source = None
        self._stop = threading.Event()
        self._thread = None

    def _tree_snapshot(self):
        try:
            root = psutil.Process(self.pid)
        except psutil.NoSuchProcess:
            return None, None, self._memory_source

        procs = [root] + root.children(recursive=True)
        total_memory = 0
        total_cpu = 0.0
        source = "rss"
        for proc in procs:
            try:
                try:
                    full = proc.memory_full_info()
                    pss = getattr(full, "pss", None)
                except (psutil.AccessDenied, AttributeError):
                    pss = None
                if pss is not None:
                    total_memory += pss
                    source = "pss"
                else:
                    total_memory += proc.memory_info().rss
                total_cpu += sum(proc.cpu_times()[:2])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total_memory, total_cpu, source

    def _run(self):
        while not self._stop.is_set():
            memory, cpu, source = self._tree_snapshot()
            if memory is not None:
                self._samples.append(memory)
                self._peak_bytes = max(self._peak_bytes, memory)
                self._memory_source = source
            if cpu is not None:
                self._cpu_times.append(cpu)
            time.sleep(self.interval)

    def start(self):
        """Start sampling in a background thread. A no-op without psutil."""
        if psutil is None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the background sampler and wait for it to exit."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 4)

    def result(self, wall_time_s=None):
        """Summarise the samples collected since start()."""
        mean = sum(self._samples) / len(self._samples) if self._samples else None
        cpu_time = self._cpu_times[-1] if self._cpu_times else None
        cpu_pct = None
        if cpu_time is not None and wall_time_s:
            cpu_pct = round(cpu_time / wall_time_s * 100.0, 2)
        return {
            "peak_memory_bytes": self._peak_bytes or None,
            "mean_memory_bytes": mean,
            "memory_source": self._memory_source,
            "sample_count": len(self._samples),
            "cpu_time_s": cpu_time,
            "cpu_utilization_pct": cpu_pct,
        }
