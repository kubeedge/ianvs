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

"""Sandbox profile and per-test-case resource limits"""

from core.common.constant import ParadigmType
from core.common.log import LOGGER

DEFAULT_BASE_IMAGE = "python:3.9-slim"
DEFAULT_TIMEOUT_S = 3600

# paradigm-aware defaults (max_memory_mb, max_cpu_cores), applied field by field when
# device_simulation does not set them; starting estimates, to be tuned with measured usage
PARADIGM_RESOURCE_DEFAULTS = {
    ParadigmType.SINGLE_TASK_LEARNING.value: (1024, 1),
    ParadigmType.INCREMENTAL_LEARNING.value: (1536, 2),
    ParadigmType.LIFELONG_LEARNING.value: (3072, 2),
    ParadigmType.FEDERATED_LEARNING.value: (2048, 4),
    ParadigmType.FEDERATED_CLASS_INCREMENTAL_LEARNING.value: (2560, 4),
    ParadigmType.JOINT_INFERENCE.value: (4096, 2),
    ParadigmType.MULTIEDGE_INFERENCE.value: (2048, 2),
}

# keys accepted now but applied by later stages (network emulation, metrics profiler)
_DEFERRED_DEVICE_KEYS = ("network_bandwidth_mbps", "network_latency_ms")


def _positive_number(name, value):
    # bool is a subclass of int, so reject it explicitly
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} ({value}) must be a positive number.")
    return value


def _str_list(name, value):
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise ValueError(f"sandbox_profile {name} ({value}) must be a list of paths.")
    return value


# pylint: disable=too-few-public-methods,too-many-instance-attributes
class SandboxProfile:
    """
    SandboxProfile: how test cases run inside the simulation cluster.

    Parameters
    ----------
    config : dict
        the sandbox_profile block of the benchmarking job, keys:
        enabled (bool), image (prebuilt image, else one is built), base_image,
        requirements (extra pip requirement files or local .whl files), data_paths (extra host paths
        the algorithm reads), timeout (seconds per test case), system_metrics.
    device_simulation : dict
        optional device_simulation block: max_memory_mb, max_cpu_cores.
    """

    def __init__(self, config, device_simulation=None):
        self.enabled = True
        self.image = ""
        self.base_image = DEFAULT_BASE_IMAGE
        self.requirements = []
        self.data_paths = []
        self.timeout = DEFAULT_TIMEOUT_S
        self.system_metrics = "all"
        self._parse_config(config)
        self.device_simulation = self._parse_device_simulation(device_simulation or {})

    def _parse_config(self, config):
        if not isinstance(config, dict):
            raise ValueError(f"sandbox_profile ({config}) must be a dict.")
        for key, value in config.items():
            if key not in self.__dict__:
                LOGGER.warning("unknown sandbox_profile key ignored: %s", key)
                continue
            self.__dict__[key] = value

        if not isinstance(self.enabled, bool):
            raise ValueError(f"sandbox_profile enabled ({self.enabled}) must be true or false.")
        for key in ("image", "base_image"):
            if not isinstance(getattr(self, key), str):
                raise ValueError(f"sandbox_profile {key} must be a string.")
        if not self.base_image:
            raise ValueError("sandbox_profile base_image must not be empty.")
        _str_list("requirements", self.requirements)
        _str_list("data_paths", self.data_paths)
        _positive_number("sandbox_profile timeout", self.timeout)

    @staticmethod
    def _parse_device_simulation(config):
        if not isinstance(config, dict):
            raise ValueError(f"device_simulation ({config}) must be a dict.")
        parsed = {}
        for key, value in config.items():
            if key in ("max_memory_mb", "max_cpu_cores"):
                parsed[key] = _positive_number(f"device_simulation {key}", value)
            elif key in _DEFERRED_DEVICE_KEYS:
                LOGGER.warning("device_simulation %s is not applied yet; "
                               "network emulation is part of a later stage", key)
            else:
                LOGGER.warning("unknown device_simulation key ignored: %s", key)
        return parsed

    def resources(self, paradigm_type):
        """
        resource limits for a test case of the paradigm: device_simulation values,
        else the paradigm default, field by field.

        Returns
        -------
        tuple
            (max_memory_mb, max_cpu_cores)
        """
        default_memory, default_cpus = PARADIGM_RESOURCE_DEFAULTS.get(paradigm_type, (2048, 2))
        return (self.device_simulation.get("max_memory_mb", default_memory),
                self.device_simulation.get("max_cpu_cores", default_cpus))
