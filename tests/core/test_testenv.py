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

"""Tests for the capability floor an example can declare in testenv.yaml."""

import sys
import types

import pytest

# aliased: pytest would otherwise try to collect `TestEnv` as a test class.
from core.testenvmanager.testenv.testenv import TestEnv as Environment


def fake_torch(available=True, capability=(8, 6), name="Fake GPU", memory_gb=16):
    """Build a torch stand-in exposing only what device_profile() reads."""
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: available,
        current_device=lambda: 0,
        get_device_capability=lambda index: capability,
        get_device_properties=lambda index: types.SimpleNamespace(
            name=name,
            total_memory=int(memory_gb * 1024 ** 3),
        ),
    )
    return torch


@pytest.fixture
def install_torch(monkeypatch):
    """Register a torch stand-in for the duration of one test."""

    def install(module):
        monkeypatch.setitem(sys.modules, "torch", module)

    return install


def build_config(**testenv):
    """A testenv config carrying only the fields _check_fields() insists on."""
    config = {"metrics": [{"name": "accuracy"}]}
    config.update(testenv)
    return {"testenv": config}


def test_declared_floor_is_parsed_from_the_config():
    env = Environment(build_config(min_compute_capability="8.0"))

    assert env.min_compute_capability == (8, 0)


def test_absent_key_leaves_no_floor():
    """39 of the 41 testenv.yaml files declare nothing; they must keep working."""
    env = Environment(build_config())

    assert env.min_compute_capability is None


def test_malformed_floor_is_rejected_at_parse_time():
    with pytest.raises(ValueError, match="min_compute_capability"):
        Environment(build_config(min_compute_capability="ampere"))


def test_inadequate_device_is_refused_before_the_dataset_is_touched(install_torch):
    """The check runs first, so nothing is downloaded for a run that cannot proceed.

    `dataset` is None here. Reaching dataset processing would raise AttributeError,
    so a RuntimeError proves the capability check ran ahead of it.
    """
    install_torch(fake_torch(capability=(6, 1), name="NVIDIA GeForce GTX 1080"))
    env = Environment(build_config(min_compute_capability="8.0"))

    with pytest.raises(RuntimeError, match="compute capability"):
        env.prepare()
