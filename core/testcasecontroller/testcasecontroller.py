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

"""Test Case Controller"""

import copy

from core.common import utils
from core.common.constant import TestObjectType
from core.testcasecontroller.algorithm import Algorithm
from core.testcasecontroller.simulation import SimulationController
from core.testcasecontroller.testcase import TestCase


class TestCaseController:
    """
    Test Case Controller:
    Control the runtime behavior of test cases like instance generation and vanish.
    """

    def __init__(self):
        self.test_cases = []

    def build_testcases(self, test_env, test_object):
        """
        Build multiple test cases by Using a test environment and multiple test algorithms.
        """

        test_object_type = test_object.get("type")
        test_object_config = test_object.get(test_object_type)
        if test_object_type == TestObjectType.ALGORITHMS.value:
            algorithms = self._parse_algorithms_config(test_object_config)
            for algorithm in algorithms:
                self.test_cases.append(TestCase(test_env, algorithm))

    def run_testcases(self, workspace, sandbox=None, simulation=None):
        """
        Run all test cases.

        Parameters
        ----------
        workspace : str
            Benchmarking job workspace.
        sandbox : SandboxConfig, optional
            When present and enabled, every test case is executed inside the
            simulation sandbox: its own worker process, its own transient
            runtime and its own CPU/memory envelope. When absent or disabled,
            execution is byte-for-byte identical to the previous behaviour.
        simulation : Simulation, optional
            Cluster topology, required only by the cluster tier.

        Returns
        -------
        (succeed_testcases, succeed_results) : tuple
        """
        if sandbox is not None and getattr(sandbox, "enabled", False):
            controller = SimulationController(
                sandbox_config=sandbox,
                simulation=simulation,
                workspace=workspace,
            )
            return controller.run_testcases(self.test_cases, workspace)

        return self._run_testcases_inline(workspace)

    def _run_testcases_inline(self, workspace):
        """
        The original, unsandboxed execution path. Unchanged.

        Kept as the default so that adding the sandbox cannot regress any of
        the existing examples: with no sandbox block configured, this is the
        code that runs.
        """
        succeed_results = {}
        succeed_testcases = []
        for testcase in self.test_cases:
            try:
                res, time = (testcase.run(workspace), utils.get_local_time())
            except Exception as err:
                raise RuntimeError(f"testcase(id={testcase.id}) runs failed, error: {err}") from err

            succeed_results[testcase.id] = (res, time)
            succeed_testcases.append(testcase)

        return succeed_testcases, succeed_results

    @classmethod
    def _parse_algorithms_config(cls, config):
        algorithms = []
        for algorithm_config in config:
            name = algorithm_config.get("name")
            config_file = algorithm_config.get("url")
            if not utils.is_local_file(config_file):
                raise RuntimeError(f"not found algorithm config file({config_file}) in local")

            try:
                config = utils.yaml2dict(config_file)
                algorithm = Algorithm(name, config)
                algorithms.append(algorithm)
            except Exception as err:
                raise RuntimeError(f"algorithm config file({config_file} is not supported, "
                                f"error: {err}") from err

        new_algorithms = []
        for algorithm in algorithms:
            for modules in algorithm.modules_list:
                new_algorithm = copy.deepcopy(algorithm)
                new_algorithm.modules = modules
                new_algorithms.append(new_algorithm)

        return new_algorithms
