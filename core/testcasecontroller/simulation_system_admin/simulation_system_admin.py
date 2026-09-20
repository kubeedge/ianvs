# Copyright 2022 The KubeEdge Authors.
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


"""simulation system admin"""

import os
import shutil
import subprocess

from core.common.log import LOGGER

SEDNA_INSTALL_URL = ("https://raw.githubusercontent.com/kubeedge/sedna"
                     "/main/scripts/installation/all-in-one.sh")
MEMORY_REQUIRE_KB = 4 * 1024 * 1024    # 4GB
CPUS_REQUIRE = 4


def _run(cmd, **kwargs):
    """run a command (list, no shell); return CompletedProcess, or None if the binary is missing."""
    try:
        return subprocess.run(cmd, check=False, **kwargs)
    except (FileNotFoundError, PermissionError):
        return None


def check_host_docker():
    """
    check whether Docker is installed and its daemon is reachable on the host.
    Raises RuntimeError with an actionable message otherwise.
    """
    if shutil.which("docker") is None:
        raise RuntimeError(
            "docker is not installed; install it from https://docs.docker.com/get-docker/")

    ret = _run(["docker", "version"], capture_output=True, text=True)
    if ret is None or ret.returncode != 0:
        raise RuntimeError(
            "docker is installed but the daemon is not reachable; is it running?")

    LOGGER.info("check docker successful")


def check_host_kind():
    """
    check whether Kind is installed on the host.
    Raises RuntimeError with an actionable message otherwise.
    """
    if shutil.which("kind") is None:
        raise RuntimeError(
            "kind is not installed; "
            "install it from https://kind.sigs.k8s.io/docs/user/quick-start/")

    ret = _run(["kind", "version"], capture_output=True, text=True)
    if ret is None or ret.returncode != 0:
        raise RuntimeError("kind is installed but `kind version` failed.")

    LOGGER.info("check Kind successful")


def _read_cgroup_limit(*paths):
    """return the first numeric cgroup limit found in paths, or None if unlimited/absent."""
    for path in paths:
        try:
            with open(path, encoding="utf-8") as limit_file:
                raw = limit_file.read().split()
        except OSError:
            continue
        if raw and raw[0].isdigit():
            return int(raw[0])
    return None


def _cgroup_cpu_limit():
    """CPU quota of the current cgroup (v2 then v1) as a float, or None if unlimited."""
    try:
        with open("/sys/fs/cgroup/cpu.max", encoding="utf-8") as cpu_max:
            quota, period = cpu_max.read().split()[:2]
        if quota != "max":
            return int(quota) / int(period)
    except (OSError, ValueError):
        pass
    quota = _read_cgroup_limit("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period = _read_cgroup_limit("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota and period:
        return quota / period
    return None


def get_host_free_memory_size():
    """
    return the currently available memory on the host (in kB).
    Uses MemAvailable (falls back to MemFree) from /proc/meminfo.
    """
    try:
        with open("/proc/meminfo", encoding="utf-8") as meminfo:
            fields = {}
            for line in meminfo:
                key, _, value = line.partition(":")
                fields[key.strip()] = value.strip()
    except OSError as err:
        raise RuntimeError(
            f"cannot read host memory info (simulation requires a Linux host): {err}") from err

    free = None
    for key in ("MemAvailable", "MemFree"):
        if key in fields:
            free = int(fields[key].split()[0])
            break
    if free is None:
        raise RuntimeError("cannot find MemAvailable/MemFree in /proc/meminfo")

    # inside a container /proc/meminfo reports the host; honour the cgroup limit too
    limit = _read_cgroup_limit("/sys/fs/cgroup/memory.max",
                               "/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if limit is not None and limit < 1 << 60:
        free = min(free, limit // 1024)
    return free


def check_host_memory():
    """
    check whether the current memory is sufficient(>=4GB)
    """
    memory_free = get_host_free_memory_size()

    if memory_free >= MEMORY_REQUIRE_KB:
        LOGGER.info("check memory successful")
    else:
        msg = (f"The current free memory is insufficient. "
               f"Current Memory Free: {memory_free} kB, Memory Require: {MEMORY_REQUIRE_KB} kB")
        LOGGER.error(msg)
        raise RuntimeError(msg)


def get_host_number_of_cpus():
    """
    return the number of cpus
    """
    if hasattr(os, "sched_getaffinity"):
        count = len(os.sched_getaffinity(0))
    else:
        count = os.cpu_count() or 0
    quota = _cgroup_cpu_limit()
    if quota is not None:
        count = min(count, int(quota))
    return count


def check_host_cpu():
    """
    check whether the number of CPUs is sufficient (>=4cores)
    """
    number_of_cpus = get_host_number_of_cpus()

    if number_of_cpus >= CPUS_REQUIRE:
        LOGGER.info("check cpu successful")
    else:
        msg = (f"The number of cpus is insufficient. "
               f"Number of Cpus: {number_of_cpus}, Cpus Require: {CPUS_REQUIRE}")
        LOGGER.error(msg)
        raise RuntimeError(msg)


def check_host_enviroment():
    """
    check the host enviroment, includes docker, kind, cpu and memory.
    """
    check_host_docker()
    check_host_kind()
    check_host_memory()
    check_host_cpu()


def _installer_env(simulation):
    """environment for the sedna installer; unset values keep the installer defaults."""
    env = dict(os.environ)
    if simulation.cluster_name:
        env["CLUSTER_NAME"] = simulation.cluster_name
    env["NUM_CLOUD_WORKER_NODES"] = str(simulation.cloud_number)
    env["NUM_EDGE_NODES"] = str(simulation.edge_number)
    if simulation.kubeedge_version:
        env["KUBEEDGE_VERSION"] = simulation.kubeedge_version
    if simulation.sedna_version:
        env["SEDNA_VERSION"] = simulation.sedna_version
    return env


def _fetch_installer():
    """download the sedna all-in-one script (fails loudly on HTTP errors)."""
    ret = _run(["curl", "-fsSL", SEDNA_INSTALL_URL], capture_output=True)
    if ret is None or ret.returncode != 0:
        raise RuntimeError(f"failed to download the installer from {SEDNA_INSTALL_URL}")
    return ret.stdout


def build_simulation_enviroment(simulation):
    """
    build a simulation enviroment
    """
    check_host_enviroment()         # check the enviroment

    ret = _run(["bash", "-s", "--"], input=_fetch_installer(), env=_installer_env(simulation))
    if ret is None or ret.returncode != 0:
        # a failed install can leave a partial cluster behind; clean up best-effort
        LOGGER.error("simulation enviroment build failed, cleaning up the partial cluster")
        try:
            destory_simulation_enviroment(simulation)
        except RuntimeError as err:
            LOGGER.error("cleanup after failed build also failed: %s", err)
        raise RuntimeError("The simulation enviroment build failed.")

    LOGGER.info("Congratulation! The simulation enviroment build successful!")


def destory_simulation_enviroment(simulation):
    """
    destroy the simulation enviroment; returns the installer's exit code.
    """
    env = dict(os.environ)
    if simulation.cluster_name:
        env["CLUSTER_NAME"] = simulation.cluster_name
    ret = _run(["bash", "-s", "--", "clean"], input=_fetch_installer(), env=env)
    return -1 if ret is None else ret.returncode


# correctly spelled aliases; the misspelled names are kept for backward compatibility
build_simulation_environment = build_simulation_enviroment
destroy_simulation_environment = destory_simulation_enviroment
check_host_environment = check_host_enviroment
