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

"""
System Metrics Profiler: resource measurements of a test case, taken from outside it.

While a simulation job runs, the profiler samples the cgroup of the job's pod on its
node (memory, CPU, block I/O), the pod's network counters and the size of its output
directory. When the job ends, it turns the samples into the system metrics of the
test case. Nothing runs inside the test case, so its behaviour is unchanged.
"""

import os
from datetime import datetime, timezone

from core.common.log import LOGGER

MB = 2 ** 20
NOT_AVAILABLE = "N/A"

# the system metrics of a test case (proposal §6.4), in leaderboard order
SYSTEM_METRIC_NAMES = (
    "wall_time_s",
    "cpu_utilization_avg", "cpu_utilization_max", "cpu_cores_used_avg",
    "peak_memory_mb", "memory_utilization_avg", "oom_failure",
    "disk_read_bytes", "disk_write_bytes", "disk_read_mb_s_avg", "disk_write_mb_s_avg",
    "disk_usage_peak_mb", "model_artifact_size_mb",
    "network_bandwidth_mbps_avg", "network_bandwidth_mbps_peak",
    "network_latency_ms", "network_profile",
    "gpu_utilization_avg", "gpu_utilization_max", "gpu_memory_peak_mb",
    "gpu_power_watts_avg", "gpu_energy_joules",
)

# GPU fields need NVIDIA's device plugin in the cluster, which the simulation cluster
# does not have; N/A is a normal result, not an error
_GPU_METRICS = ("gpu_utilization_avg", "gpu_utilization_max", "gpu_memory_peak_mb",
                "gpu_power_watts_avg", "gpu_energy_joules")

# files a test case saves as models
MODEL_FILE_SUFFIXES = (".pth", ".pt", ".ckpt", ".pb", ".h5", ".hdf5", ".onnx", ".pkl",
                       ".pickle", ".joblib", ".model", ".bin", ".safetensors", ".tflite",
                       ".zip", ".npz", ".index", ".meta")


def network_profile(bandwidth_mbps, latency_ms):
    """the label of a network condition, e.g. 50mbps-20ms (proposal §6.3)."""
    return f"{bandwidth_mbps:g}mbps-{latency_ms:g}ms"


def parse_sample(text):
    """
    a sample printed by SimulationJobAdministrator.sample_pod: key=value lines.
    Missing values (e.g. a file that is gone) are left out.
    """
    sample = {}
    for line in text.splitlines():
        key, _, value = line.partition("=")
        fields = value.split()
        if not fields:
            continue
        try:
            numbers = [float(field) for field in fields]
        except ValueError:
            continue
        if key in ("io", "net") and len(numbers) == 2:
            sample[f"{key}_in"], sample[f"{key}_out"] = numbers
        elif len(numbers) == 1:
            sample[key] = numbers[0]
    return sample


def model_artifact_size_mb(directory):
    """the total size of the model files under a directory."""
    total = 0
    for base, _, files in os.walk(directory):
        for name in files:
            stem = name.lower()
            if stem.endswith(MODEL_FILE_SUFFIXES) or ".data-" in stem:
                try:
                    total += os.path.getsize(os.path.join(base, name))
                except OSError:
                    continue
    return round(total / MB, 3)


def directory_size_mb(directory):
    """the total size of the files under a directory."""
    total = 0
    for base, _, files in os.walk(directory):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(base, name))
            except OSError:
                continue
    return round(total / MB, 3)


def _rfc3339(value):
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError):
        return None


class SystemMetricsProfiler:
    """
    samples one simulation job and computes its system metrics.

    Parameters
    ----------
    resources : tuple
        (max_memory_mb, max_cpu_cores) of the job's pod
    network : tuple
        (bandwidth_mbps, latency_ms) emulated for the job
    """

    def __init__(self, resources, network):
        self.memory_limit_mb, self.cpu_limit = resources
        self.bandwidth_mbps, self.latency_ms = network
        self.samples = []
        # the node's time when the test case could start (its network was ready)
        self.started = None

    def add(self, sample):
        """record a sample (see parse_sample); samples without a time are dropped."""
        if "t" in sample:
            self.samples.append(sample)

    def _series(self, key):
        return [(s["t"], s[key]) for s in self.samples if key in s]

    def _last(self, key, default=None):
        series = self._series(key)
        return series[-1][1] if series else default

    @staticmethod
    def _peak_rate(series, scale):
        """the highest rate between consecutive samples of a counter, times scale."""
        peak = 0.0
        for (t_start, v_start), (t_end, v_end) in zip(series, series[1:]):
            if t_end > t_start and v_end >= v_start:
                peak = max(peak, (v_end - v_start) / (t_end - t_start) * scale)
        return peak

    @staticmethod
    def _per_second(value, wall, digits=4):
        return round(value / wall, digits) if wall else 0.0

    def _cpu(self, wall):
        cpu_seconds = (self._last("cpu", 0.0) or 0.0) / 1e6
        average = self._per_second(cpu_seconds, wall * self.cpu_limit, 6)
        series = self._series("cpu")
        peak = self._peak_rate(series, 1e-6 / self.cpu_limit) if len(series) > 1 else average
        return {
            "cpu_utilization_avg": round(min(average, 1.0), 4),
            "cpu_utilization_max": round(min(max(peak, average), 1.0), 4),
            "cpu_cores_used_avg": self._per_second(cpu_seconds, wall),
        }

    def _memory(self, oom):
        memory = [v for _, v in self._series("mem")]
        peak = max(memory + [v for _, v in self._series("peak")], default=0.0)
        if oom:
            # the kill happens at the limit; a sample may not have caught the spike
            peak = max(peak, self.memory_limit_mb * MB)
        average = sum(memory) / len(memory) if memory else 0.0
        return {
            "peak_memory_mb": round(peak / MB, 3),
            "memory_utilization_avg": round(average / (self.memory_limit_mb * MB), 4),
            "oom_failure": bool(oom),
        }

    def _disk(self, wall, output_dir):
        read_bytes = self._last("io_in", 0.0) or 0.0
        write_bytes = self._last("io_out", 0.0) or 0.0
        usage_peak = max(max((v for _, v in self._series("du")), default=0.0) / 1024,
                         directory_size_mb(output_dir) if output_dir else 0.0)
        return {
            "disk_read_bytes": int(read_bytes),
            "disk_write_bytes": int(write_bytes),
            "disk_read_mb_s_avg": self._per_second(read_bytes / MB, wall),
            "disk_write_mb_s_avg": self._per_second(write_bytes / MB, wall),
            "disk_usage_peak_mb": round(usage_peak, 3),
            "model_artifact_size_mb": model_artifact_size_mb(output_dir) if output_dir else 0.0,
        }

    def _network(self, wall):
        traffic = [(s["t"], s.get("net_in", 0.0) + s.get("net_out", 0.0))
                   for s in self.samples if "net_in" in s or "net_out" in s]
        bits = (traffic[-1][1] - traffic[0][1]) * 8 if len(traffic) > 1 else 0.0
        return {
            "network_bandwidth_mbps_avg": self._per_second(bits / 1e6, wall),
            "network_bandwidth_mbps_peak": round(self._peak_rate(traffic, 8 / 1e6), 4),
            "network_latency_ms": self.latency_ms,
            "network_profile": network_profile(self.bandwidth_mbps, self.latency_ms),
        }

    def wall_time(self, terminated, reported=None):
        """
        the test case's wall time: as the runner measured it (monotonic, excludes the
        start-up); for a test case killed before it could report, from when its network
        was ready to the container's finish; else the container's lifetime.
        """
        if reported:
            return float(reported)
        finish = _rfc3339(terminated.get("finishedAt"))
        if self.started is not None and finish:
            elapsed = finish.replace(tzinfo=timezone.utc).timestamp() - self.started
            if elapsed > 0:
                return elapsed
        return self.container_span(terminated)

    def container_span(self, terminated):
        """the container's lifetime: its start and finish times, else the samples' span."""
        start = _rfc3339(terminated.get("startedAt"))
        finish = _rfc3339(terminated.get("finishedAt"))
        if start and finish and finish > start:
            return (finish - start).total_seconds()
        if len(self.samples) > 1:
            return self.samples[-1]["t"] - self.samples[0]["t"]
        return 0.0

    def metrics(self, terminated, oom, output_dir, reported_wall_time=None):
        """
        the system metrics of the job.

        Parameters
        ----------
        terminated : dict
            the terminated state of the pod's container
        oom : bool
            whether the test case was killed for exceeding its memory limit
        output_dir : string
            the test case's output directory, copied back to the host
        reported_wall_time : float
            the wall time the runner measured, if it reported one
        """
        wall = self.wall_time(terminated, reported_wall_time)
        # CPU, disk and network counters cover the whole container (interpreter
        # start-up and imports included), so their rates use the container's lifetime
        span = max(wall, self.container_span(terminated))
        result = {"wall_time_s": round(wall, 3)}
        result.update(self._cpu(span))
        result.update(self._memory(oom))
        result.update(self._disk(span, output_dir))
        result.update(self._network(span))
        result.update({name: NOT_AVAILABLE for name in _GPU_METRICS})
        if not self.samples:
            LOGGER.warning("no resource samples of the simulation job; its CPU, memory, "
                           "disk and network metrics are 0")
        return result


def select_metrics(metrics, selection):
    """the system metrics the sandbox profile asks for: "all" or a list of names."""
    if selection == "all":
        return dict(metrics)
    return {name: metrics[name] for name in selection if name in metrics}
