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
import subprocess

from core.common.log import LOGGER


def check_host_docker():
    """
    check whether Docker is installed on the host.
    If Docker is not installed, try to install Docker with one-click installation script.
    """

    shell_cmd = "docker version | head -n 2"
    check_docker = subprocess.run(shell_cmd, shell=True, check=True)

    if check_docker.returncode != 0:
        # trying to install docker
        LOGGER.info("trying to install docker")
        try:
            shell_install_docker = "curl -fsSL https://get.docker.com | \
bash -s docker --mirror Aliyun"
            install_docker = subprocess.run(
                shell_install_docker, shell=True, check=True)

            if install_docker.returncode == 0:
                LOGGER.info("successfully installed docker")
            else:
                raise RuntimeError("install docker failed")
        except Exception as err:
            raise RuntimeError(f"install docker failed, error: {err}.") from err

    LOGGER.info("check docker successful")


def check_host_kind():
    """
    check whether Kind is installed on the host.
    If Kind is not installed, try to install Kind with one-click installation script.
    """

    shell_cmd = "kind version"
    check_kind = subprocess.run(shell_cmd, shell=True, check=True)

    if check_kind.returncode == 0:
        LOGGER.info("check Kind successful")
    else:
        try:
            shell_install_kind = "curl -Lo ./kind \
https://kind.sigs.k8s.io/dl/v0.17.0/kind-linux-amd64 && \
chmod +x ./kind && mv ./kind /usr/local/bin/kind"
            install_kind = subprocess.run(
                shell_install_kind, shell=True, check=True)

            if install_kind.returncode == 0:
                LOGGER.info("successfully installed kind")
            else:
                LOGGER.exception("install kind failed")
                raise RuntimeError("install kind failed")
        except Exception as err:
            raise RuntimeError(f"install kind failed, error: {err}.") from err


def _read_first_line(path):
    """return the stripped first line of path, or None if it cannot be read"""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.readline().strip()
    except OSError:
        return None


def get_host_free_memory_size():
    """
    return the memory this process may actually still use(in kB)

    /proc/meminfo is not namespaced by cgroups, so in a container it reports
    the host's memory rather than the container's limit. Read the cgroup
    limit first and fall back to /proc/meminfo only when no limit applies.
    """
    # cgroup v2: "max" means no limit is set
    limit = _read_first_line("/sys/fs/cgroup/memory.max")
    usage = _read_first_line("/sys/fs/cgroup/memory.current")
    if limit and limit != "max" and usage:
        return max(0, (int(limit) - int(usage)) // 1024)

    # cgroup v1: an unset limit is a very large sentinel rather than a flag
    limit = _read_first_line("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    usage = _read_first_line("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    if limit and usage and int(limit) < 2 ** 62:
        return max(0, (int(limit) - int(usage)) // 1024)

    # no cgroup limit applies, so the host figure is the effective one.
    # MemAvailable, not MemFree: MemFree excludes reclaimable page cache and
    # therefore understates what is usable on any warm host.
    with subprocess.Popen("grep MemAvailable /proc/meminfo", shell=True,
                          stdout=subprocess.PIPE) as get_memory_info:
        memory_info = get_memory_info.stdout.read().decode("utf-8")
        return int(memory_info.split(":")[1].strip().split(" ")[0])


def check_host_memory():
    """
    check whether the current memory is sufficient(>=4GB)

    """
    memory_free = get_host_free_memory_size()
    memory_require = 4 * 1024 * 1024    # 4GB

    if memory_free >= memory_require:
        LOGGER.info("check memory successful")
    else:
        LOGGER.exception(
            "The current free memory is insufficient. \
Current Memory Free: %s kB, Memory Require: %s kB",
            memory_free, memory_require)
        raise RuntimeError("The current free memory is insufficient.")


def get_host_number_of_cpus():
    """
    return the number of cpus this process may actually use

    lscpu reports the host's cpus regardless of any cgroup cpu quota, so in a
    container it overstates what is available. Prefer the cgroup quota, then
    the scheduler affinity mask, then the host cpu count.
    """
    # cgroup v2: "<quota> <period>", or "max <period>" when no quota is set
    cpu_max = _read_first_line("/sys/fs/cgroup/cpu.max")
    if cpu_max:
        quota, _, period = cpu_max.partition(" ")
        if quota != "max" and period:
            return max(1, int(int(quota) / int(period)))

    # cgroup v1: a quota of -1 means unlimited
    quota = _read_first_line("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period = _read_first_line("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota and period and int(quota) > 0:
        return max(1, int(int(quota) / int(period)))

    # no quota applies, so the affinity mask is the next most accurate figure.
    # it is Linux only, so fall back to the host cpu count where it is absent
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))

    return max(1, os.cpu_count() or 1)


def check_host_cpu():
    """
    check whether the number of CPUs is sufficient (>=4cores)

    """
    number_of_cpus = get_host_number_of_cpus()
    cpus_require = 4

    if number_of_cpus >= cpus_require:
        LOGGER.info("check cpu successful")
    else:
        LOGGER.info(
            "The number of cpus is insufficient. Number of Cpus: %s kB, Cpus Require: %s kB",
            number_of_cpus, cpus_require)
        raise RuntimeError("The number os cpus is insufficient.")


def check_host_enviroment():
    """
    check the host enviroment, includes docker, kind, cpu and memory.

    """
    check_host_docker()
    check_host_kind()
    check_host_memory()
    check_host_cpu()


def build_simulation_enviroment(simulation):
    """
    build a simulation enviroment

    """

    check_host_enviroment()         # check the enviroment

    shell_cmd = "curl https://raw.githubusercontent.com/kubeedge/sedna\
/master/scripts/installation/all-in-one.sh | " \
        f"NUM_CLOUD_WORKER_NODES={simulation.cloud_number} " \
        f"NUM_EDGE_NODES={simulation.edge_number} " \
        f"KUBEEDGE_VERSION={simulation.kubeedge_version} " \
        f"SEDNA_VERSION={simulation.sedna_version} " \
        f"CLUSTER_NAME={simulation.cluster_name} bash -"

    build_simulation_env_ret = subprocess.run(
        shell_cmd, shell=True, check=True)

    if build_simulation_env_ret.returncode == 0:
        LOGGER.info(
            "Congratulation! The simulation enviroment build successful!")
    else:
        raise RuntimeError("The simulation enviroment build failed.")


def destory_simulation_enviroment(simulation):
    """
    build the simulation enviroment

    """
    shell_cmd = "curl https://raw.githubusercontent.com/kubeedge/sedna\
/main/scripts/installation/all-in-one.sh | " \
        f"CLUSTER_NAME={simulation.cluster_name} bash /dev/stdin clean"

    retcode = subprocess.call(shell_cmd, shell=True)

    return retcode
