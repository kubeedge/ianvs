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

"""Tests for gpu visibility handling in ParadigmBase.

These cover the two defects reported in #765:

1. `use_gpu: false` was a silent no-op, so a run configured for cpu inherited
   whatever devices the host exposed.
2. `use_gpu: true` was applied after the algorithm module had already been
   instantiated, so AI frameworks had picked their device list before the
   restriction took effect.

Both are ordering-sensitive, so every case below asserts the value of
CUDA_VISIBLE_DEVICES *at the moment the algorithm module is instantiated*
rather than after construction has finished. Asserting afterwards would pass
even against the original buggy code.
"""

import os

import pytest

from core.testcasecontroller.algorithm.paradigm.base import ParadigmBase


MULTI_GPU_HOST = "0,1,2,3"


class RecordingModule:
    """Stands in for an algorithm module.

    `_get_module_instances()` calls `get_module_instance()` on every configured
    module; for a real run that is the point where basemodel.py is imported and
    instantiated, and therefore the point where a framework reads
    CUDA_VISIBLE_DEVICES. Recording the variable here is what makes these tests
    sensitive to ordering.
    """

    def __init__(self):
        self.visibility_at_instantiation = "not called"

    def get_module_instance(self, module_type):
        self.visibility_at_instantiation = os.environ.get("CUDA_VISIBLE_DEVICES")
        return module_type


def build_paradigm(tmp_path, **kwargs):
    """Construct a ParadigmBase and report what the module saw."""
    module = RecordingModule()
    ParadigmBase(str(tmp_path), modules={"basemodel": module}, **kwargs)
    return module.visibility_at_instantiation


@pytest.fixture(autouse=True)
def isolate_cuda_visible_devices(monkeypatch):
    """Keep each case from leaking CUDA_VISIBLE_DEVICES into the next."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


def test_use_gpu_false_hides_every_device_on_a_multi_gpu_host(tmp_path, monkeypatch):
    """The original no-op: cpu-only runs must not inherit the host's devices."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", MULTI_GPU_HOST)

    assert build_paradigm(tmp_path, use_gpu=False) == "-1"


def test_use_gpu_true_restricts_to_one_device_before_modules_load(tmp_path, monkeypatch):
    """The original ordering defect: the module saw all four devices."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", MULTI_GPU_HOST)

    assert build_paradigm(tmp_path, use_gpu=True) == "0"


def test_unset_use_gpu_leaves_an_existing_value_alone(tmp_path, monkeypatch):
    """Most testenv.yaml files omit the key; their behaviour must not change."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", MULTI_GPU_HOST)

    assert build_paradigm(tmp_path) == MULTI_GPU_HOST


def test_unset_use_gpu_does_not_introduce_the_variable(tmp_path):
    """Omitting the key must not create CUDA_VISIBLE_DEVICES where none existed."""
    assert build_paradigm(tmp_path) is None
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


@pytest.mark.parametrize(
    "use_gpu, expected",
    [
        (False, "-1"),
        (True, "0"),
        (None, MULTI_GPU_HOST),
    ],
)
def test_set_gpu_visibility_maps_each_state(monkeypatch, use_gpu, expected):
    """The three states of the setting, independent of paradigm construction."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", MULTI_GPU_HOST)

    ParadigmBase._set_gpu_visibility(use_gpu)  # pylint: disable=protected-access

    assert os.environ["CUDA_VISIBLE_DEVICES"] == expected


def test_each_construction_applies_the_setting_rather_than_inheriting_it(tmp_path, monkeypatch):
    """A job builds one paradigm per algorithm, all from a single TestEnv.

    `TestCaseController.build_testcases()` passes the same `test_env` object to
    every `TestCase`, so cases in one run share a `use_gpu` value. What matters
    across constructions is therefore that each one applies that value itself
    instead of trusting whatever the previous construction left in the
    environment. Raised in review of #890.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", MULTI_GPU_HOST)
    assert build_paradigm(tmp_path, use_gpu=False) == "-1"

    # something outside the paradigm restores the host's devices
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", MULTI_GPU_HOST)

    assert build_paradigm(tmp_path, use_gpu=False) == "-1"


def test_repeated_construction_is_stable_for_the_gpu_path(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", MULTI_GPU_HOST)

    assert build_paradigm(tmp_path, use_gpu=True) == "0"
    assert build_paradigm(tmp_path, use_gpu=True) == "0"
