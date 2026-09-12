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

"""SimulationController: runs every test case inside a sandbox tier.

Fault containment and teardown are both driven from a ``finally`` block:

B14  the original ``run_testcases`` loop re-raised out of the loop on the
     first exception, discarding every result already computed. Here a
     failed test case is recorded and the run continues, unless
     ``sandbox.fail_fast`` is set.
B7   teardown (``sandbox.teardown()`` and ``env_admin.destroy()``) always
     runs, including when a test case raises with ``fail_fast`` set.
"""

from core.common import utils
from core.common.constant import SandboxMode
from core.common.log import LOGGER
from core.testcasecontroller.simulation.env_admin import (
    SimulationEnvironmentAdministrator,
)
from core.testcasecontroller.simulation.profiler import ProfileResult
from core.testcasecontroller.simulation.sandbox.process import ProcessSandbox

_METRIC_COLUMN_MAP = {
    "peak_memory": "peak_memory_mb",
    "mean_memory": "mean_memory_mb",
    "cpu_utilization": "cpu_utilization_pct",
    "cpu_time": "cpu_time_s",
    "wall_time": "wall_time_s",
}


# pylint: disable=too-few-public-methods
class SimulationController:
    """Runs a job's test cases inside the simulation sandbox."""

    def __init__(self, sandbox_config, simulation, workspace):
        self.sandbox_config = sandbox_config
        self.simulation = simulation
        self.workspace = workspace
        self.env_admin = SimulationEnvironmentAdministrator(sandbox_config, simulation)

    def _make_sandbox(self, mode, ianvs_root):
        if mode == SandboxMode.PROCESS.value:
            return ProcessSandbox(self.sandbox_config, ianvs_root)
        # pylint: disable=import-outside-toplevel
        from core.testcasecontroller.simulation.sandbox.cluster import ClusterSandbox
        backend = self.env_admin.build()
        return ClusterSandbox(self.sandbox_config, ianvs_root, backend)

    def run_testcases(self, test_cases, workspace):
        """
        Run every test case inside the resolved sandbox tier.

        A crashing, hanging or OOM-killed test case is recorded as a failed
        result rather than aborting the run: every prior result is
        preserved regardless of what happens to a later test case, unless
        ``sandbox.fail_fast`` is set. Returns the same
        ``(succeed_testcases, succeed_results)`` shape as the unsandboxed
        path, so ``BenchmarkingJob.run()`` needs no further changes.
        """
        mode = self.env_admin.resolve_mode()
        sandbox = self._make_sandbox(mode, ianvs_root=".")
        succeed_testcases = []
        succeed_results = {}

        try:
            sandbox.prepare()
            for testcase in test_cases:
                result, profile = sandbox.run_testcase(testcase, workspace)

                if not profile.succeeded:
                    LOGGER.error(
                        "testcase(id=%s) failed inside the sandbox: %s",
                        testcase.id, profile.error)
                    if self.sandbox_config.fail_fast:
                        raise RuntimeError(
                            f"testcase(id={testcase.id}) failed inside the "
                            f"sandbox and fail_fast is set: {profile.error}")
                    continue

                merged = self._merge_metrics(result, profile)
                succeed_results[testcase.id] = (merged, utils.get_local_time())
                succeed_testcases.append(testcase)
        finally:
            sandbox.teardown()
            self.env_admin.destroy()

        return succeed_testcases, succeed_results

    def _merge_metrics(self, result, profile: ProfileResult):
        """
        Merge selected sandbox system metrics into the paradigm's own
        result dict, so ``StoryManager`` sees one flat dict and needs no
        changes of its own.
        """
        if not isinstance(result, dict):
            return result

        merged = dict(result)
        profile_dict = profile.as_dict()
        selected = self.sandbox_config.metrics
        if not selected:
            return merged
        if "all" in selected:
            merged.update(profile_dict)
            return merged

        for name in selected:
            column = _METRIC_COLUMN_MAP.get(name, name)
            if column in profile_dict:
                merged[column] = profile_dict[column]
        return merged
