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

"""
Regression tests for the simulation environment teardown bug.

Bug: BenchmarkingJob.run() called build_simulation_enviroment() but never
called the matching destory_simulation_enviroment(), so every simulation
run (success or failure) leaked its kind cluster.

These tests mock build_simulation_enviroment / destory_simulation_enviroment
directly (they shell out to docker/kind/curl) so the control-flow contract
of run() can be verified without a real docker+kind host.
"""

import unittest
from unittest.mock import MagicMock, patch

from core.cmd.obj.benchmarkingjob import BenchmarkingJob


def _make_job(simulation=None):
    """Build a BenchmarkingJob without going through config-file parsing."""
    job = BenchmarkingJob.__new__(BenchmarkingJob)
    job.name = "test-job"
    job.workspace = "./workspace"
    job.test_object = {}
    job.simulation = simulation
    job.test_env = MagicMock()
    job.testcase_controller = MagicMock()
    job.testcase_controller.run_testcases.return_value = ([], [])
    job.rank = MagicMock()
    return job


class TestSimulationTeardown(unittest.TestCase):
    """Verify destory_simulation_enviroment is always paired with a build."""

    @patch("core.cmd.obj.benchmarkingjob.destory_simulation_enviroment")
    @patch("core.cmd.obj.benchmarkingjob.build_simulation_enviroment")
    def test_teardown_called_after_successful_run(self, mock_build, mock_destroy):
        """Happy path: build -> run testcases -> teardown, in that order."""
        simulation = MagicMock(cluster_name="ianvs-test")
        job = _make_job(simulation)

        job.run()

        mock_build.assert_called_once_with(simulation)
        mock_destroy.assert_called_once_with(simulation)

    @patch("core.cmd.obj.benchmarkingjob.destory_simulation_enviroment")
    @patch("core.cmd.obj.benchmarkingjob.build_simulation_enviroment")
    def test_teardown_called_even_when_testcases_raise(self, mock_build, mock_destroy):
        """
        This is the core of the bug: before the fix, an exception anywhere
        between build and the end of run() skipped teardown entirely, and
        even a clean run never called it. The original error must still
        propagate -- teardown must not swallow it.
        """
        simulation = MagicMock(cluster_name="ianvs-test")
        job = _make_job(simulation)
        job.testcase_controller.run_testcases.side_effect = RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            job.run()

        mock_build.assert_called_once_with(simulation)
        mock_destroy.assert_called_once_with(simulation)

    @patch("core.cmd.obj.benchmarkingjob.destory_simulation_enviroment")
    @patch("core.cmd.obj.benchmarkingjob.build_simulation_enviroment")
    def test_no_simulation_configured_skips_both(self, mock_build, mock_destroy):
        """When no `simulation:` block is configured, neither should fire."""
        job = _make_job(simulation=None)

        job.run()

        mock_build.assert_not_called()
        mock_destroy.assert_not_called()

    @patch("core.cmd.obj.benchmarkingjob.destory_simulation_enviroment")
    @patch("core.cmd.obj.benchmarkingjob.build_simulation_enviroment")
    def test_teardown_skipped_when_build_itself_fails(self, mock_build, mock_destroy):
        """
        If build never finished (e.g. host env check failed), there is
        nothing to tear down -- destroy should not fire against a cluster
        that was never created.
        """
        simulation = MagicMock(cluster_name="ianvs-test")
        job = _make_job(simulation)
        mock_build.side_effect = RuntimeError("host environment check failed")

        with self.assertRaises(RuntimeError):
            job.run()

        mock_build.assert_called_once_with(simulation)
        mock_destroy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
