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
from core.testcasecontroller.testcase import TestCase


class TestCaseController:
    """
    Test Case Controller:
    Control the runtime behavior of test cases like instance generation and vanish.
    """

    def __init__(self):
        self.test_cases = []

    def build_testcases(self, test_env, algorithms):
        """
        Build multiple test cases by Using a test environment and multiple test algorithms.
        """
        for algorithm in algorithms:
            for modules in algorithm.modules_list:
                new_algorithm = copy.deepcopy(algorithm)
                new_algorithm.modules = modules
                self.test_cases.append(TestCase(test_env, new_algorithm))

    def run_testcases(self, workspace):
        """
        Run all test cases.
        """
        succeed_results = {}
        succeed_testcases = []
        for testcase in self.test_cases:
            try:
                res, time = (testcase.run(workspace), utils.get_local_time())
            except Exception as err:
                raise Exception(f"testcase(id={testcase.id}) runs failed, error: {err}") from err

            succeed_results[testcase.id] = (res, time)
            succeed_testcases.append(testcase)

        return succeed_testcases, succeed_results
