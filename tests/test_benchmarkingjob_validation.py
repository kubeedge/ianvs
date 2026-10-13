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

"""Tests for BenchmarkingJob validation, Algorithm validation, and get_visualization_func."""

import sys
import os
import unittest

# Ensure the repo root is on sys.path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_job_config(**overrides):
    """Return a minimal BenchmarkingJob config dict, applying any overrides."""
    cfg = {
        "name": "test-job",
        "workspace": "./workspace",
        "test_object": {"type": "algorithms", "algorithms": [{"name": "a", "url": "/nonexistent.yaml"}]},
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# BenchmarkingJob validation tests
# ---------------------------------------------------------------------------

class TestBenchmarkingJobValidation(unittest.TestCase):
    """Unit tests for BenchmarkingJob._check_fields()."""

    def _make_job(self, config):
        """Import lazily to avoid top-level import side-effects."""
        from core.cmd.obj.benchmarkingjob import BenchmarkingJob
        return BenchmarkingJob(config)

    def test_missing_name_raises(self):
        cfg = _minimal_job_config(name="")
        with self.assertRaises(ValueError, msg="empty name should raise ValueError"):
            self._make_job(cfg)

    def test_wrong_type_name_raises(self):
        cfg = _minimal_job_config(name=42)
        with self.assertRaises(ValueError, msg="integer name should raise ValueError"):
            self._make_job(cfg)

    def test_wrong_type_workspace_raises(self):
        cfg = _minimal_job_config(workspace=123)
        with self.assertRaises(ValueError, msg="integer workspace should raise ValueError"):
            self._make_job(cfg)

    def test_wrong_type_test_object_raises(self):
        cfg = _minimal_job_config(test_object="not-a-dict")
        with self.assertRaises(ValueError, msg="string test_object should raise ValueError"):
            self._make_job(cfg)

    def test_missing_testenv_raises(self):
        """Config with no testenv key → test_env stays None → ValueError."""
        cfg = _minimal_job_config()
        # test_env and rank are set only via 'testenv' / 'rank' YAML keys,
        # which are not in this dict.  _check_fields must catch both.
        with self.assertRaises((ValueError, RuntimeError)):
            self._make_job(cfg)

    def test_missing_rank_raises(self):
        """Even if test_object is valid, absent rank should raise."""
        cfg = _minimal_job_config()
        with self.assertRaises((ValueError, RuntimeError)):
            self._make_job(cfg)


# ---------------------------------------------------------------------------
# Algorithm validation tests
# ---------------------------------------------------------------------------

class TestAlgorithmValidation(unittest.TestCase):
    """Unit tests for Algorithm._check_fields()."""

    def _make_algorithm(self, name, paradigm_type):
        from core.testcasecontroller.algorithm.algorithm import Algorithm
        config = {
            "algorithm": {
                "paradigm_type": paradigm_type,
                "modules": [],
            }
        }
        return Algorithm(name, config)

    def test_empty_name_raises(self):
        with self.assertRaises(ValueError):
            self._make_algorithm("", "singletasklearning")

    def test_non_string_name_raises(self):
        with self.assertRaises(ValueError):
            self._make_algorithm(42, "singletasklearning")

    def test_empty_paradigm_raises(self):
        with self.assertRaises(ValueError):
            self._make_algorithm("algo", "")

    def test_invalid_paradigm_raises(self):
        with self.assertRaises(ValueError):
            self._make_algorithm("algo", "unsupported_paradigm")

    def test_invalid_fl_data_setting_type_raises(self):
        from core.testcasecontroller.algorithm.algorithm import Algorithm
        config = {
            "algorithm": {
                "paradigm_type": "singletasklearning",
                "fl_data_setting": "invalid_not_a_dict",
                "modules": [],
            }
        }
        with self.assertRaises(ValueError) as ctx:
            Algorithm("algo", config)
        self.assertIn("fl_data_setting", str(ctx.exception))

    def test_invalid_fl_data_partition_raises(self):
        from core.testcasecontroller.algorithm.algorithm import Algorithm
        config = {
            "algorithm": {
                "paradigm_type": "singletasklearning",
                "fl_data_setting": {"data_partition": "invalid_partition"},
                "modules": [],
            }
        }
        with self.assertRaises(ValueError) as ctx:
            Algorithm("algo", config)
        self.assertIn("data_partition", str(ctx.exception))

    def test_invalid_fl_non_iid_ratio_raises(self):
        from core.testcasecontroller.algorithm.algorithm import Algorithm
        for bad_ratio in (0, -0.5, 1.5, "not-a-number", True, False):
            config = {
                "algorithm": {
                    "paradigm_type": "singletasklearning",
                    "fl_data_setting": {"non_iid_ratio": bad_ratio},
                    "modules": [],
                }
            }
            with self.assertRaises(ValueError) as ctx:
                Algorithm("algo", config)
            self.assertIn("non_iid_ratio", str(ctx.exception))


# ---------------------------------------------------------------------------
# get_visualization_func tests
# ---------------------------------------------------------------------------

class TestGetVisualizationFunc(unittest.TestCase):
    """Unit tests for get_visualization_func() guard."""

    def setUp(self):
        self.mod = __import__(
            "core.storymanager.visualization.visualization",
            fromlist=["get_visualization_func", "print_table"]
        )

    def test_valid_mode_returns_callable(self):
        func = self.mod.get_visualization_func("print_table")
        self.assertTrue(callable(func))

    def test_invalid_mode_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            self.mod.get_visualization_func("plot_table")
        self.assertIn("plot_table", str(ctx.exception))
        self.assertIn("Valid options", str(ctx.exception))

    def test_empty_mode_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.mod.get_visualization_func("")

    def test_arbitrary_attr_does_not_leak(self):
        """Ensure arbitrary module attributes cannot be accessed via get_visualization_func."""
        with self.assertRaises(ValueError):
            self.mod.get_visualization_func("sys")


if __name__ == "__main__":
    unittest.main()
