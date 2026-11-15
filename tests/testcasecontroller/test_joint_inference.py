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

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock

# Ensure sedna mocks are available if sedna is not installed in the test environment
mock_sedna = MagicMock()
for mod in [
    "sedna",
    "sedna.core",
    "sedna.core.incremental_learning",
    "sedna.core.lifelong_learning",
    "sedna.core.joint_inference",
    "sedna.core.base",
]:
    if mod not in sys.modules:
        sys.modules[mod] = mock_sedna

# Safely load paradigm base and joint inference modules without full package tree side-effects
repo_root = Path(__file__).resolve().parents[2]

fed_path = repo_root / "core" / "testcasecontroller" / "algorithm" / "paradigm" / "sedna_federated_learning.py"
if "core.testcasecontroller.algorithm.paradigm.sedna_federated_learning" not in sys.modules:
    spec_fed = importlib.util.spec_from_file_location(
        "core.testcasecontroller.algorithm.paradigm.sedna_federated_learning",
        fed_path,
    )
    mod_fed = importlib.util.module_from_spec(spec_fed)
    sys.modules["core.testcasecontroller.algorithm.paradigm.sedna_federated_learning"] = mod_fed
    spec_fed.loader.exec_module(mod_fed)

base_path = repo_root / "core" / "testcasecontroller" / "algorithm" / "paradigm" / "base.py"
if "core.testcasecontroller.algorithm.paradigm.base" not in sys.modules:
    spec_base = importlib.util.spec_from_file_location(
        "core.testcasecontroller.algorithm.paradigm.base",
        base_path,
    )
    mod_base = importlib.util.module_from_spec(spec_base)
    sys.modules["core.testcasecontroller.algorithm.paradigm.base"] = mod_base
    spec_base.loader.exec_module(mod_base)

ji_path = repo_root / "core" / "testcasecontroller" / "algorithm" / "paradigm" / "joint_inference" / "joint_inference.py"
if "core.testcasecontroller.algorithm.paradigm.joint_inference.joint_inference" not in sys.modules:
    spec_ji = importlib.util.spec_from_file_location(
        "core.testcasecontroller.algorithm.paradigm.joint_inference.joint_inference",
        ji_path,
    )
    mod_ji = importlib.util.module_from_spec(spec_ji)
    sys.modules["core.testcasecontroller.algorithm.paradigm.joint_inference.joint_inference"] = mod_ji
    spec_ji.loader.exec_module(mod_ji)
else:
    mod_ji = sys.modules["core.testcasecontroller.algorithm.paradigm.joint_inference.joint_inference"]

JointInference = mod_ji.JointInference


class TestJointInferenceCleanupAndRun(unittest.TestCase):
    def setUp(self):
        self.mock_edge_instance = MagicMock()
        self.mock_cloud_instance = MagicMock()

        self.mock_edge_module = MagicMock()
        self.mock_edge_module.get_module_instance.return_value = self.mock_edge_instance

        self.mock_cloud_module = MagicMock()
        self.mock_cloud_module.get_module_instance.return_value = self.mock_cloud_instance

        self.modules = {
            "edgemodel": self.mock_edge_module,
            "cloudmodel": self.mock_cloud_module,
        }

    def test_cleanup_mining_free_mode_without_mining_attribute(self):
        """In mining-free mode, job does not have hard_example_mining_algorithm.
        _cleanup must complete safely without raising AttributeError."""
        ji = JointInference(
            workspace="/tmp/test_workspace",
            modules=self.modules,
            hard_example_mining_mode="mining-free",
        )

        class MiningFreeJob:
            pass

        job = MiningFreeJob()
        # Should not raise AttributeError: 'MiningFreeJob' object has no attribute 'hard_example_mining_algorithm'
        ji._cleanup(job)

        self.mock_edge_instance.cleanup.assert_called_once()
        self.mock_cloud_instance.cleanup.assert_called_once()

    def test_cleanup_with_mining_algorithm(self):
        """When hard_example_mining_algorithm is present and has cleanup(), it must be called."""
        ji = JointInference(
            workspace="/tmp/test_workspace",
            modules=self.modules,
            hard_example_mining_mode="mining-then-inference",
        )

        mock_mining_instance = MagicMock()

        class MiningJob:
            def __init__(self, mining_algo):
                self.hard_example_mining_algorithm = mining_algo

        job = MiningJob(mock_mining_instance)
        ji._cleanup(job)

        mock_mining_instance.cleanup.assert_called_once()
        self.mock_edge_instance.cleanup.assert_called_once()
        self.mock_cloud_instance.cleanup.assert_called_once()

    def test_cleanup_with_mining_algorithm_no_cleanup_method(self):
        """When hard_example_mining_algorithm is present but has no cleanup(), handle safely."""
        ji = JointInference(
            workspace="/tmp/test_workspace",
            modules=self.modules,
        )

        class MiningAlgoWithoutCleanup:
            pass

        class MiningJob:
            def __init__(self):
                self.hard_example_mining_algorithm = MiningAlgoWithoutCleanup()

        job = MiningJob()
        ji._cleanup(job)

        self.mock_edge_instance.cleanup.assert_called_once()
        self.mock_cloud_instance.cleanup.assert_called_once()

    def test_cleanup_module_without_cleanup_method(self):
        """Module instances without cleanup() method are safely ignored."""
        no_cleanup_instance = object()
        module = MagicMock()
        module.get_module_instance.return_value = no_cleanup_instance

        ji = JointInference(
            workspace="/tmp/test_workspace",
            modules={"custom": module},
        )

        class MockJob:
            pass

        ji._cleanup(MockJob())

    def test_run_guarantees_cleanup_on_inference_exception(self):
        """When _inference raises an exception, _cleanup must still be executed."""
        ji = JointInference(
            workspace="/tmp/test_workspace",
            modules=self.modules,
        )
        ji.set_config = MagicMock()
        mock_job = MagicMock()
        ji.build_paradigm_job = MagicMock(return_value=mock_job)
        ji._cleanup = MagicMock()
        ji._inference = MagicMock(side_effect=RuntimeError("Simulated inference failure"))

        with self.assertRaises(RuntimeError) as context:
            ji.run()

        self.assertEqual(str(context.exception), "Simulated inference failure")
        ji._cleanup.assert_called_once_with(mock_job)

    def test_run_success_invokes_cleanup(self):
        """When _inference succeeds, results are returned and _cleanup is executed."""
        ji = JointInference(
            workspace="/tmp/test_workspace",
            modules=self.modules,
        )
        ji.set_config = MagicMock()
        mock_job = MagicMock()
        ji.build_paradigm_job = MagicMock(return_value=mock_job)
        ji._cleanup = MagicMock()
        ji._inference = MagicMock(return_value=["sample_output_1", "sample_output_2"])
        ji.system_metric_info = {"latency": 12.5}

        result, metric_info = ji.run()

        self.assertEqual(result, ["sample_output_1", "sample_output_2"])
        self.assertEqual(metric_info, {"latency": 12.5})
        ji._cleanup.assert_called_once_with(mock_job)


if __name__ == "__main__":
    unittest.main()
