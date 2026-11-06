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

"""Configuration for the simulation sandbox: SandboxConfig, ResourceQuota
and the parse_memory() helper for Kubernetes-style memory quantities.
"""

import os
import re

from core.common.constant import IsolationLevel, SandboxMode

_BINARY_UNITS = {
    "Ki": 2 ** 10, "Mi": 2 ** 20, "Gi": 2 ** 30,
    "Ti": 2 ** 40, "Pi": 2 ** 50, "Ei": 2 ** 60,
}
_DECIMAL_UNITS = {
    "KB": 10 ** 3, "MB": 10 ** 6, "GB": 10 ** 9,
    "TB": 10 ** 12, "PB": 10 ** 15, "EB": 10 ** 18,
}
_QUANTITY_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]*)\s*$")


def parse_memory(value):
    """
    Parse a Kubernetes-style memory quantity into a byte count.

    Accepts binary suffixes (``Gi``, ``Mi``, ``Ki``, ...), decimal suffixes
    with a trailing ``B`` (``GB``, ``MB``, ``KB``, ...), a bare int/float
    (interpreted as bytes) or a numeric string with no suffix. ``None`` and
    ``''`` mean "unspecified" and return ``None`` rather than raising, so an
    absent quota is not an error.
    """
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"memory quantity({value!r}) must not be a boolean.")
    if isinstance(value, (int, float)):
        return int(value)

    match = _QUANTITY_RE.match(str(value))
    if not match:
        raise ValueError(f"memory quantity({value!r}) is not a valid quantity.")

    number, unit = match.groups()
    number = float(number)

    if not unit:
        return int(number)
    if unit in _BINARY_UNITS:
        return int(number * _BINARY_UNITS[unit])
    if unit in _DECIMAL_UNITS:
        return int(number * _DECIMAL_UNITS[unit])

    raise ValueError(
        f"memory quantity({value!r}) has an unrecognised unit({unit!r}); "
        f"expected one of {sorted(_BINARY_UNITS)} (binary) or "
        f"{sorted(_DECIMAL_UNITS)} (decimal).")


class ResourceQuota:
    """
    A CPU/memory/timeout ceiling applied to one sandboxed test case.

    Parameters
    ----------
    memory : str, int, float or None
        A Kubernetes-style quantity, e.g. ``"2Gi"``, or a plain byte count.
    cpus : int, float or None
        Core count; fractional values are allowed. Rejected if it exceeds
        the host's own CPU count, since a quota larger than the host
        constrains nothing.
    timeout : int or None
        Wall-clock seconds before the worker is sent SIGTERM, then SIGKILL.
    """

    def __init__(self, memory=None, cpus=None, timeout=None):
        self.memory_bytes = parse_memory(memory)
        self.cpus = self._check_cpus(cpus)
        self.timeout = timeout

    @staticmethod
    def _check_cpus(cpus):
        if cpus is None or cpus == "":
            return None
        if isinstance(cpus, bool):
            raise ValueError(f"cpus({cpus!r}) must not be a boolean.")
        cpus = float(cpus)
        if cpus <= 0:
            raise ValueError(f"cpus({cpus}) must be positive.")
        host_cpus = os.cpu_count() or 1
        if cpus > host_cpus:
            raise ValueError(
                f"cpus({cpus}) exceeds the host CPU count({host_cpus}); a "
                "quota larger than the host constrains nothing.")
        return cpus

    def is_unbounded(self):
        """True when neither a memory nor a CPU ceiling is set."""
        return self.memory_bytes is None and self.cpus is None

    def as_dict(self):
        """A flat, JSON-safe representation of this quota."""
        return {
            "memory_bytes": self.memory_bytes,
            "cpus": self.cpus,
            "timeout": self.timeout,
        }


_SANDBOX_FIELDS = {
    "enabled", "mode", "isolation", "resources", "metrics",
    "fail_fast", "keep_workdir",
}
_RESOURCE_FIELDS = {"memory", "cpus", "timeout"}
_VALID_MODES = {mode.value for mode in SandboxMode}
_VALID_ISOLATION = {level.value for level in IsolationLevel}


# pylint: disable=too-few-public-methods
class SandboxConfig:
    """
    The parsed ``sandbox`` block of a ``benchmarkingjob.yaml``.

    The whole block is optional. ``SandboxConfig()`` / ``SandboxConfig.disabled()``
    both produce a disabled config, which is the default: with no ``sandbox``
    block, ``TestCaseController`` takes the original, unsandboxed code path.
    """

    def __init__(self, sandbox_config=None):
        self.enabled = False
        self.mode = SandboxMode.PROCESS.value
        self.isolation = IsolationLevel.VENV.value
        self.fail_fast = False
        self.keep_workdir = False
        self.metrics = []
        self.quota = ResourceQuota()

        if sandbox_config:
            self._parse_config(sandbox_config)

    @classmethod
    def disabled(cls):
        """An explicitly disabled sandbox config."""
        return cls()

    def _parse_config(self, sandbox_config):
        unknown = set(sandbox_config) - _SANDBOX_FIELDS
        if unknown:
            raise ValueError(
                f"sandbox config has unknown field(s) {sorted(unknown)}; "
                f"expected one of {sorted(_SANDBOX_FIELDS)}.")

        if "mode" in sandbox_config:
            mode = sandbox_config["mode"]
            if mode not in _VALID_MODES:
                raise ValueError(
                    f"sandbox mode({mode!r}) is not supported; expected "
                    f"one of {sorted(_VALID_MODES)}.")
            self.mode = mode

        if "isolation" in sandbox_config:
            isolation = sandbox_config["isolation"]
            if isolation not in _VALID_ISOLATION:
                raise ValueError(
                    f"sandbox isolation({isolation!r}) is not supported; "
                    f"expected one of {sorted(_VALID_ISOLATION)}.")
            self.isolation = isolation

        if "resources" in sandbox_config:
            resources = sandbox_config["resources"]
            unknown_res = set(resources) - _RESOURCE_FIELDS
            if unknown_res:
                raise ValueError(
                    f"sandbox resources has unknown field(s) "
                    f"{sorted(unknown_res)}; expected one of "
                    f"{sorted(_RESOURCE_FIELDS)}.")
            self.quota = ResourceQuota(**resources)

        if "metrics" in sandbox_config:
            self.metrics = list(sandbox_config["metrics"])

        for flag in ("enabled", "fail_fast", "keep_workdir"):
            if flag in sandbox_config:
                setattr(self, flag, bool(sandbox_config[flag]))
