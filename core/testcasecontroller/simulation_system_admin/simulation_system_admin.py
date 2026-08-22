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
    Check whether Kind is installed on the host.
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


def get_host_free_memory_size():
    """
    Return the current memory(free) on the host(in kB)
    """
    shell_cmd = "cat /proc/meminfo | grep MemFree"   # in kB
    with subprocess.Popen(shell_cmd, shell=True, stdout=subprocess.PIPE) as get_memory_info:
        memory_info = get_memory_info.stdout.read()
        memory_free = int(str(memory_info).split(":")[1].strip().split(" ")[0])
        return memory_free


def check_host_memory():
    """
    Check whether the current memory is sufficient(>=4GB)

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
    Return the number of cpus

    """
    shell_cmd = "lscpu | grep CPU:"
    with subprocess.Popen(shell_cmd, shell=True, stdout=subprocess.PIPE) as get_cpu_info:
        cpu_info = get_cpu_info.stdout.read()
        number_of_cpus = int(str(cpu_info).split(":")[
                             1].strip().split("\\")[0])
        return number_of_cpus


def check_host_cpu():
    """
    Check whether the number of CPUs is sufficient (>=4cores)

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
    Check the host environment, includes docker, kind, cpu and memory.

    """
    check_host_docker()
    check_host_kind()
    check_host_memory()
    check_host_cpu()


def build_simulation_enviroment(simulation):
    """
    Build a simulation environment

    """
    check_host_enviroment()  # check the environment

    shell_cmd = (
        "curl https://raw.githubusercontent.com/kubeedge/sedna\
/master/scripts/installation/all-in-one.sh | "
        f"NUM_CLOUD_WORKER_NODES={simulation.cloud_number} "
        f"NUM_EDGE_NODES={simulation.edge_number} "
        f"KUBEEDGE_VERSION={simulation.kubeedge_version} "
        f"SEDNA_VERSION={simulation.sedna_version} "
        f"CLUSTER_NAME={simulation.cluster_name} bash -"
    )

    build_simulation_env_ret = subprocess.run(
        shell_cmd, shell=True, check=True)

    if build_simulation_env_ret.returncode == 0:
        LOGGER.info(
            "Congratulation! The simulation environment build successful!")
    else:
        raise RuntimeError("The simulation environment build failed.")


def destory_simulation_enviroment(simulation):
    """
    Build the simulation environment

    """
    shell_cmd = (
        "curl https://raw.githubusercontent.com/kubeedge/sedna\
/main/scripts/installation/all-in-one.sh | "
        f"CLUSTER_NAME={simulation.cluster_name} bash /dev/stdin clean"
    )

    retcode = subprocess.call(shell_cmd, shell=True)

    return retcode
