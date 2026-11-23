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

"""Compute device capability reporting.

Ianvs decides device *visibility* in `ParadigmBase` from the `use_gpu` setting.
That answers whether a GPU is used. It does not answer whether the GPU can
execute what an algorithm is about to ask of it, which is a separate question
with its own failure mode: on hardware below compute capability 8.0, PyTorch
emulates `bfloat16` instead of refusing it, so a benchmark completes and reports
numbers produced by emulated arithmetic.

This module reports the facts an algorithm needs to make that decision. It
returns plain data rather than torch objects, so that `core` keeps no torch
dependency of its own.

See issue #888.
"""

from core.common.log import LOGGER

BF16_MIN_MAJOR = 8
"""First CUDA compute capability major version with native bfloat16 (Ampere)."""


def device_profile():
    """Report the resolved compute device and what it can natively execute.

    torch is imported lazily because it is not a dependency of `core`; a
    checkout without torch installed gets the cpu profile rather than an
    ImportError.

    Returns
    ------
    dict
        device: str
            "cuda" or "cpu".
        name: str or None
            Device name as reported by the driver, None on cpu.
        capability: tuple of (int, int) or None
            CUDA compute capability, None on cpu.
        bf16_native: bool
            Whether bfloat16 executes natively rather than through emulation.
        total_memory_gb: float or None
            Device memory in GiB, None on cpu.

    """
    # pylint: disable=import-outside-toplevel
    # `core` does not depend on torch; see the module docstring.
    try:
        import torch
    except ImportError:
        return _cpu_profile()

    if not torch.cuda.is_available():
        return _cpu_profile()

    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    major, minor = torch.cuda.get_device_capability(index)

    return {
        "device": "cuda",
        "name": properties.name,
        "capability": (major, minor),
        "bf16_native": major >= BF16_MIN_MAJOR,
        "total_memory_gb": round(properties.total_memory / 1024 ** 3, 2),
    }


def _cpu_profile():
    """Profile reported when no CUDA device is usable."""
    return {
        "device": "cpu",
        "name": None,
        "capability": None,
        "bf16_native": False,
        "total_memory_gb": None,
    }


def parse_capability(value):
    """Normalise a declared compute capability into a (major, minor) pair.

    Accepts what a yaml author is likely to write: "8.0", "8.6", 8.0 or 8.

    Parameters
    ---------
    value: str or float or int
        the declared capability.

    Returns
    ------
    tuple of (int, int)

    """
    text = str(value).strip()
    parts = text.split(".")

    well_formed = (
        1 <= len(parts) <= 2
        and parts[0].isdigit()
        and (len(parts) == 1 or parts[1] == "" or parts[1].isdigit())
    )
    if not well_formed:
        raise ValueError(
            f"testenv min_compute_capability(value={value}) must look like "
            f'"8.0" or "7.5".'
        )

    major = int(parts[0])
    minor = int(parts[1]) if len(parts) == 2 and parts[1] != "" else 0
    return (major, minor)


def check_min_capability(required):
    """Refuse a run whose device cannot meet a declared capability floor.

    Only a cuda device below the floor is an error. A run resolving to cpu is
    reported and allowed to continue, because an example may legitimately be
    exercised on cpu — `use_gpu: false` asks for exactly that.

    Parameters
    ---------
    required: tuple of (int, int) or None
        the declared floor; None means the example did not declare one.

    """
    if required is None:
        return

    profile = device_profile()

    if profile["device"] != "cuda":
        LOGGER.warning(
            "testenv declares min_compute_capability %d.%d but no cuda device was "
            "resolved; continuing on cpu.",
            required[0], required[1],
        )
        return

    if profile["capability"] < required:
        major, minor = profile["capability"]
        raise RuntimeError(
            f"testenv requires compute capability {required[0]}.{required[1]}, but "
            f"{profile['name']} is {major}.{minor}. Results from this device would not "
            f"be comparable with the declared configuration."
        )


def supports_bf16():
    """Report whether bfloat16 executes natively on the resolved device.

    Deliberately derived from the compute capability rather than from
    `torch.cuda.is_bf16_supported()`. That function defaults to
    `including_emulation=True` and so answers True on hardware with no native
    bfloat16, which is the trap this module exists to avoid.

    Warns every time the answer is False on a CUDA device. Deliberately not
    suppressed after the first occurrence: a benchmarking job runs its test cases
    in one process, so a once-per-process warning would announce the first
    affected measurement and silently pass over the rest — the same defect this
    module exists to remove. One warning per affected model load is proportionate,
    since each corresponds to a measurement the caveat applies to.

    Returns
    ------
    bool

    """
    profile = device_profile()
    if profile["device"] == "cuda" and not profile["bf16_native"]:
        major, minor = profile["capability"]
        LOGGER.warning(
            "%s is compute capability %d.%d; bfloat16 has no native path below "
            "%d.0 and would be emulated. Falling back to float32. Timings "
            "collected in bfloat16 on this device would not describe the hardware.",
            profile["name"], major, minor, BF16_MIN_MAJOR,
        )
        return False

    return profile["bf16_native"]
