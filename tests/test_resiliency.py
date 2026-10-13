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

"""Unit tests for core benchmark execution resiliency and stability fixes (Issue #610)."""

import os
import shutil
import tempfile
import uuid
import unittest
from unittest.mock import MagicMock

from core.testcasecontroller.testcase import TestCase
from core.testcasecontroller.testcasecontroller import TestCaseController
from core.storymanager.rank import Rank
from core.common import utils


class TestResiliency(unittest.TestCase):
    """Test suite for Issue #610 resiliency defects."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_output_dir_no_infinite_loop(self):
        """Test TestCase._get_output_dir does not loop infinitely when path exists."""
        algorithm_mock = MagicMock()
        algorithm_mock.name = "test_algo"

        test_case = TestCase(test_env=MagicMock(), algorithm=algorithm_mock)

        # Create pre-existing directory matching initial output_dir
        initial_dir = os.path.join(self.temp_dir, algorithm_mock.name, str(test_case.id))
        os.makedirs(initial_dir, exist_ok=True)

        output_dir = test_case._get_output_dir(self.temp_dir)

        self.assertNotEqual(output_dir, initial_dir)
        self.assertTrue(output_dir.startswith(initial_dir))
        self.assertFalse(os.path.exists(output_dir))

    def test_partial_failure_retention_in_testcase_controller(self):
        """Test TestCaseController.run_testcases retains successful testcases when one fails."""
        controller = TestCaseController()

        case_ok = MagicMock()
        case_ok.id = uuid.uuid4()
        case_ok.run.return_value = {"accuracy": 95.0}

        case_fail = MagicMock()
        case_fail.id = uuid.uuid4()
        case_fail.run.side_effect = RuntimeError("Simulated testcase failure")

        controller.test_cases = [case_ok, case_fail]

        succeed_testcases, succeed_results = controller.run_testcases(self.temp_dir)

        self.assertEqual(len(succeed_testcases), 1)
        self.assertEqual(succeed_testcases[0].id, case_ok.id)
        self.assertIn(case_ok.id, succeed_results)
        self.assertNotIn(case_fail.id, succeed_results)

    def test_rank_save_mode_validation_and_draw_pictures_guard(self):
        """Test Rank save_mode allows str and _draw_pictures handles None Matrix safely."""
        config = {
            "sort_by": [{"accuracy": "descend"}],
            "save_mode": "selected_and_all_and_picture",
        }
        rank = Rank(config)
        self.assertEqual(rank.save_mode, "selected_and_all_and_picture")

        case_mock = MagicMock()
        case_mock.id = uuid.uuid4()
        case_mock.output_dir = self.temp_dir

        # Scalar result without "Matrix" key (returns None for .get("Matrix"))
        test_results = {case_mock.id: ({"accuracy": 95.0}, "2026-08-06 10:00:00")}

        # Should execute safely without throwing AttributeError: 'NoneType' object has no attribute 'keys'
        try:
            rank._draw_pictures([case_mock], test_results)
        except Exception as err:
            self.fail(f"_draw_pictures raised unexpected exception: {err}")

    def test_isolated_module_loading(self):
        """Test py2dict and load_module load dynamically without polluting sys.path."""
        py_file = os.path.join(self.temp_dir, "test_config.py")
        with open(py_file, "w", encoding="utf-8") as f:
            f.write("PARAM = 42\n")

        initial_sys_path = list(os.sys.path)

        res_dict = utils.py2dict(py_file)
        self.assertEqual(res_dict.get("PARAM"), 42)
        self.assertEqual(os.sys.path, initial_sys_path)

        mod = utils.load_module(py_file)
        self.assertEqual(getattr(mod, "PARAM"), 42)
        self.assertEqual(os.sys.path, initial_sys_path)


if __name__ == "__main__":
    unittest.main()
