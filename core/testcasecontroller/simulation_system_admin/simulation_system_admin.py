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

    check_docker = subprocess.run(
        ["docker", "version"], capture_output=True, check=False)

    if check_docker.returncode != 0:
        # trying to install docker
        LOGGER.info("trying to install docker")
        with subprocess.Popen(
                ["curl", "-fsSL", "https://get.docker.com"],
                stdout=subprocess.PIPE) as curl_proc:
            try:
                subprocess.run(
                    ["bash", "-s", "docker", "--mirror", "Aliyun"],
                    stdin=curl_proc.stdout, check=True)
            except subprocess.CalledProcessError as err:
                raise RuntimeError(f"install docker failed, error: {err}.") from err

        LOGGER.info("successfully installed docker")

    LOGGER.info("check docker successful")


def check_host_kind():
    """
    check whether Kind is installed on the host.
    If Kind is not installed, try to install Kind with one-click installation script.
    """

    check_kind = subprocess.run(
        ["kind", "version"], capture_output=True, check=False)

    if check_kind.returncode == 0:
        LOGGER.info("check Kind successful")
        return

    try:
        subprocess.run(
            ["curl", "-Lo", "./kind",
             "https://kind.sigs.k8s.io/dl/v0.17.0/kind-linux-amd64"], check=True)
        subprocess.run(["chmod", "+x", "./kind"], check=True)
        subprocess.run(["mv", "./kind", "/usr/local/bin/kind"], check=True)
    except subprocess.CalledProcessError as err:
        raise RuntimeError(f"install kind failed, error: {err}.") from err

    LOGGER.info("successfully installed kind")


def get_host_free_memory_size():
    """
    return the current memory(free) on the host(in kB)
    """
    shell_cmd = "cat /proc/meminfo | grep MemFree"   # in kB
    with subprocess.Popen(shell_cmd, shell=True, stdout=subprocess.PIPE) as get_memory_info:
        memory_info = get_memory_info.stdout.read()
        memory_free = int(str(memory_info).split(":")[1].strip().split(" ")[0])
        return memory_free


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
    return the number of cpus

    """
    shell_cmd = "lscpu | grep CPU:"
    with subprocess.Popen(shell_cmd, shell=True, stdout=subprocess.PIPE) as get_cpu_info:
        cpu_info = get_cpu_info.stdout.read()
        number_of_cpus = int(str(cpu_info).split(":")[
                             1].strip().split("\\")[0])
        return number_of_cpus


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

    install_env = os.environ.copy()
    install_env.update({
        "NUM_CLOUD_WORKER_NODES": str(simulation.cloud_number),
        "NUM_EDGE_NODES": str(simulation.edge_number),
        "KUBEEDGE_VERSION": str(simulation.kubeedge_version),
        "SEDNA_VERSION": str(simulation.sedna_version),
        "CLUSTER_NAME": str(simulation.cluster_name),
    })

    with subprocess.Popen(
            ["curl", "https://raw.githubusercontent.com/kubeedge/sedna"
                     "/master/scripts/installation/all-in-one.sh"],
            stdout=subprocess.PIPE) as curl_proc:
        try:
            subprocess.run(["bash", "-"], stdin=curl_proc.stdout,
                           env=install_env, check=True)
        except subprocess.CalledProcessError as err:
            raise RuntimeError("The simulation enviroment build failed.") from err

    LOGGER.info("Congratulation! The simulation enviroment build successful!")


def destory_simulation_enviroment(simulation):
    """
    build the simulation enviroment

    """
    destroy_env = os.environ.copy()
    destroy_env["CLUSTER_NAME"] = str(simulation.cluster_name)

    with subprocess.Popen(
            ["curl", "https://raw.githubusercontent.com/kubeedge/sedna"
                     "/main/scripts/installation/all-in-one.sh"],
            stdout=subprocess.PIPE) as curl_proc:
        result = subprocess.run(["bash", "/dev/stdin", "clean"],
                                stdin=curl_proc.stdout, env=destroy_env, check=False)

    return result.returncode
