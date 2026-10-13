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

"""Host capability probing for the simulation sandbox.

Fixes two verified defects from the 2022 implementation:

B1  ``check_host_docker``/``check_host_kind`` called
    ``subprocess.run(..., check=True)`` and then tested
    ``ret.returncode != 0`` — but ``check=True`` raises
    ``CalledProcessError`` on any non-zero exit, so that branch could never
    run. A host without Docker crashed with a raw traceback instead of
    taking the documented fallback.
B5  ``get_host_number_of_cpus`` shelled out to ``lscpu | grep CPU:`` and
    split the output on ``:`` and ``\\``; the field is locale-dependent and
    ``lscpu`` is absent from slim container images.
"""

import os
import platform
import shutil
import subprocess


def command_version(binary, version_flag="--version"):
    """
    Return a command's version-flag output, or ``None`` if it isn't runnable.

    Never raises: a missing binary, a non-zero exit, a timeout, or a binary
    that doesn't understand the flag all just report as unavailable.
    """
    path = shutil.which(binary)
    if path is None:
        return None
    try:
        result = subprocess.run(
            [path, version_flag], capture_output=True, text=True,
            timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return (result.stdout or result.stderr).strip() or None


def get_cpu_count():
    """
    Number of CPUs usable by this process.

    Prefers the process's own affinity mask (correct under a cgroup/cpuset
    ceiling); falls back to ``os.cpu_count()``, which is always available.
    """
    if hasattr(os, "sched_getaffinity"):
        try:
            return len(os.sched_getaffinity(0))
        except OSError:
            pass
    return os.cpu_count() or 1


def get_free_memory_bytes():
    """Free host memory in bytes, or ``None`` when it cannot be determined."""
    try:
        import psutil  # pylint: disable=import-outside-toplevel
        return psutil.virtual_memory().available
    except ImportError:
        pass

    meminfo = "/proc/meminfo"
    if not os.path.isfile(meminfo):
        return None
    with open(meminfo, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return None


def has_cgroup_v2_write_access(path="/sys/fs/cgroup"):
    """Whether this process can write cgroup v2 controllers under ``path``."""
    controllers = os.path.join(path, "cgroup.controllers")
    return os.path.isfile(controllers) and os.access(path, os.W_OK)


def check_process_tier():
    """
    The process tier needs nothing beyond a working Python interpreter.

    B15 (see ``docs/proposals/simulation/sandbox-engine/ianvs-simulation-sandbox.md``):
    the 2022 host check demanded Docker and ``kind`` unconditionally, which
    blocked macOS and CI users who only wanted single-process isolation. The
    process tier must come up on any host that can run Ianvs at all.
    """
    return True


def check_cluster_tier():
    """
    Probe whether this host can plausibly run the cluster tier.

    Returns a dict of individual checks rather than a bool, so a caller can
    report exactly what is missing instead of one opaque failure.
    """
    free_memory = get_free_memory_bytes()
    return {
        "docker": command_version("docker") is not None,
        "kind": command_version("kind") is not None,
        "kubectl": command_version("kubectl", "version") is not None,
        "linux": platform.system() == "Linux",
        "memory_ok": free_memory is None or free_memory >= 4 * 2 ** 30,
        "free_memory_bytes": free_memory,
    }


def cluster_tier_ready(checks=None):
    """True when every check required for the cluster tier passed."""
    checks = checks if checks is not None else check_cluster_tier()
    return all([
        checks["docker"], checks["kind"], checks["kubectl"],
        checks["linux"], checks["memory_ok"],
    ])
