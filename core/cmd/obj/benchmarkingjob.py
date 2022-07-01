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

"""BenchmarkingJob"""

import os

from core.testcasecontroller.algorithm import Algorithm
from core.storymanager.rank import Rank
from core.common import utils
from core.testcasecontroller.testcasecontroller import TestCaseController
from core.testenvmanager.testenv import TestEnv

# pylint: disable=too-few-public-methods
class BenchmarkingJob:
    """
    BenchmarkingJob:
        providing a end-to-end benchmarking job.

    Parameters
    ----------
    config: dict
        config of a end-to-end benchmarking job,
        includes: test env, algorithms, rank setting, etc.

    """

    def __init__(self, config):
        self.name: str = ""
        self.workspace: str = "./workspace"
        self.algorithms: list = []
        self.rank = None
        self.test_env = None
        self.testcase_controller = TestCaseController()
        self._parse_config(config)

    def _check_fields(self):
        if not self.name and not isinstance(self.name, str):
            ValueError(f"algorithm name({self.name}) must be provided and be string type.")
            raise ValueError(f"benchmarkingjob's name({self.name}) must be provided"
                             f" and be string type.")

        if not isinstance(self.workspace, str):
            raise ValueError(f"benchmarkingjob's workspace({self.workspace}) must be string type.")

    def run(self):
        """
        run a end-to-end benchmarking job,
        includes prepare test env,
                 run all test cases,
                 save results of all test cases,
                 plot the results according to the visualization config of rank.
        """
        self.workspace = os.path.join(self.workspace, self.name)

        self.test_env.prepare()

        self.testcase_controller.build_testcases(test_env=self.test_env, algorithms=self.algorithms)

        succeed_testcases, test_results = self.testcase_controller.run_testcases(self.workspace)

        if test_results:
            self.rank.save(succeed_testcases, test_results, output_dir=self.workspace)
            self.rank.plot()

    def _parse_config(self, config: dict):
        # pylint: disable=C0103
        for k, v in config.items():
            if k == str.lower(TestEnv.__name__):
                self._parse_testenv_config(v)
            elif k == str.lower(Algorithm.__name__ + "s"):
                self._parse_algorithms_config(v)
            elif k == str.lower(Rank.__name__):
                self._parse_rank_config(v)
            else:
                if k in self.__dict__:
                    self.__dict__[k] = v

        self._check_fields()

    def _parse_testenv_config(self, config_file):
        if not utils.is_local_file(config_file):
            raise Exception(f"not found testenv config file({config_file}) in local")

        try:
            config = utils.yaml2dict(config_file)
            self.test_env = TestEnv(config)
        except Exception as err:
            raise Exception(f"testenv config file({config_file}) is not supported, "
                            f"error: {err}") from err

    def _parse_rank_config(self, config):
        self.rank = Rank(config)

    def _parse_algorithms_config(self, config):
        self.algorithms = []
        for algorithm_config in config:
            name = algorithm_config.get("name")
            config_file = algorithm_config.get("url")
            if not utils.is_local_file(config_file):
                raise Exception(f"not found algorithm config file({config_file}) in local")

            try:
                config = utils.yaml2dict(config_file)
                algorithm = Algorithm(name, config)
                self.algorithms.append(algorithm)
            except Exception as err:
                raise Exception(f"algorithm config file({config_file} is not supported, "
                                f"error: {err}") from err
