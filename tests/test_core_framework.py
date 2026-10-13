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

"""Unit tests for core framework stability and resiliency fixes."""

import os
import sys
import tempfile
import uuid
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.common import utils
from core.testcasecontroller.testcase.testcase import TestCase
from core.testcasecontroller.testcasecontroller import TestCaseController
from core.storymanager.rank.rank import Rank


class MockAlgorithm:
    def __init__(self, name="mock_algo"):
        self.name = name


class MockTestCase:
    def __init__(self, tc_id, should_fail=False):
        self.id = tc_id
        self.should_fail = should_fail
        self.algorithm = MockAlgorithm()

    def run(self, workspace):
        if self.should_fail:
            raise RuntimeError(f"Mock failure for {self.id}")
        return {"accuracy": 0.95}


def test_get_output_dir_avoid_infinite_loop():
    algorithm = MockAlgorithm("test_algo")
    testcase = TestCase(test_env=None, algorithm=algorithm)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Pre-create the initial output dir to simulate collision
        initial_dir = os.path.join(tmp_dir, algorithm.name, str(testcase.id))
        os.makedirs(initial_dir, exist_ok=True)

        res_dir = testcase._get_output_dir(tmp_dir)
        assert res_dir != initial_dir
        assert res_dir.startswith(initial_dir)
        assert "_1" in res_dir


def test_run_testcases_partial_failure_retention():
    controller = TestCaseController()
    tc1 = MockTestCase(uuid.uuid4(), should_fail=False)
    tc2 = MockTestCase(uuid.uuid4(), should_fail=True)
    tc3 = MockTestCase(uuid.uuid4(), should_fail=False)

    controller.test_cases = [tc1, tc2, tc3]

    with tempfile.TemporaryDirectory() as tmp_dir:
        succeed_tcs, succeed_results = controller.run_testcases(tmp_dir)
        assert len(succeed_tcs) == 2
        assert len(succeed_results) == 2
        assert tc1 in succeed_tcs
        assert tc3 in succeed_tcs
        assert tc2 not in succeed_tcs


def test_run_testcases_all_failed_raises_runtime_error():
    controller = TestCaseController()
    tc1 = MockTestCase(uuid.uuid4(), should_fail=True)
    tc2 = MockTestCase(uuid.uuid4(), should_fail=True)
    controller.test_cases = [tc1, tc2]

    with tempfile.TemporaryDirectory() as tmp_dir:
        with pytest.raises(RuntimeError) as exc_info:
            controller.run_testcases(tmp_dir)
        assert "All testcases failed" in str(exc_info.value)


def test_rank_save_mode_str_validation():
    config = {
        "sort_by": [{"accuracy": "descend"}],
        "visualization": {"mode": "selected_only", "method": "print_table"},
        "selected_dataitem": {
            "paradigms": ["all"],
            "modules": ["all"],
            "hyperparameters": ["all"],
            "metrics": ["all"],
        },
        "save_mode": "selected_and_all"
    }

    rank = Rank(config)
    assert rank.save_mode == "selected_and_all"


def test_rank_draw_pictures_none_matrix_guard():
    config = {
        "sort_by": [{"accuracy": "descend"}],
        "visualization": {"mode": "selected_only", "method": "print_table"},
        "selected_dataitem": {
            "paradigms": ["all"],
            "modules": ["all"],
            "hyperparameters": ["all"],
            "metrics": ["all"],
        },
        "save_mode": "selected_and_all"
    }
    rank = Rank(config)

    tc = MockTestCase(uuid.uuid4())
    tc.output_dir = "/tmp/test"
    test_results = {tc.id: ({"accuracy": 0.95, "Matrix": None}, "2026-08-01 00:00:00")}

    # Should not raise AttributeError when matrix is None
    rank._draw_pictures([tc], test_results)


def test_py2dict_and_load_module():
    with tempfile.TemporaryDirectory() as tmp_dir:
        py_file = os.path.join(tmp_dir, "sample_config.py")
        with open(py_file, "w", encoding="utf-8") as f:
            f.write("KEY_A = 'VALUE_A'\nKEY_B = 123\n")

        res_dict = utils.py2dict(py_file)
        assert res_dict.get("KEY_A") == 'VALUE_A'
        assert res_dict.get("KEY_B") == 123

        mod = utils.load_module(py_file)
        assert getattr(mod, "KEY_A") == 'VALUE_A'


def test_load_module_failure_cleans_sys_path():
    with tempfile.TemporaryDirectory() as tmp_dir:
        missing_module = os.path.join(tmp_dir, "non_existent_module.py")
        orig_sys_path = list(sys.path)
        with pytest.raises(RuntimeError):
            utils.load_module(missing_module)
        assert sys.path == orig_sys_path


def test_load_module_caching_and_collision():
    with tempfile.TemporaryDirectory() as tmp_dir:
        dir1 = os.path.join(tmp_dir, "algo1")
        dir2 = os.path.join(tmp_dir, "algo2")
        os.makedirs(dir1)
        os.makedirs(dir2)

        file1 = os.path.join(dir1, "basemodel.py")
        file2 = os.path.join(dir2, "basemodel.py")

        with open(file1, "w", encoding="utf-8") as f:
            f.write("EXEC_COUNT = 1\nMODEL_VAL = 'model_1'\n")

        with open(file2, "w", encoding="utf-8") as f:
            f.write("EXEC_COUNT = 1\nMODEL_VAL = 'model_2'\n")

        # Load file1
        mod1_a = utils.load_module(file1)
        assert mod1_a.MODEL_VAL == "model_1"

        # Load file1 again -> should return cached mod1_a
        mod1_b = utils.load_module(file1)
        assert mod1_b is mod1_a

        # Load file2 with same basename -> should reload to model_2
        mod2 = utils.load_module(file2)
        assert mod2.MODEL_VAL == "model_2"


def test_clean_hardware_resources():
    # Calling _clean_hardware_resources should execute without error
    TestCaseController._clean_hardware_resources()
