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

from core.common import utils
from core.common.constant import TestObjectType
from core.testenvmanager.testenv import TestEnv
from core.storymanager.rank import Rank
from core.testcasecontroller.testcasecontroller import TestCaseController


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
        self.test_object: dict = {}
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

        if not self.test_object and not isinstance(self.test_object, dict):
            raise ValueError(f"benchmarkingjob's test_object({self.test_object})"
                             f" must be dict type.")

        test_object_types = [e.value for e in TestObjectType.__members__.values()]
        test_object_type = self.test_object.get("type")
        if test_object_type not in test_object_types:
            raise ValueError(
                f"benchmarkingjob' test_object doesn't support the type({test_object_type}), "
                f"the following test object types can be selected: {test_object_types}.")

        if not self.test_object.get(test_object_type):
            raise ValueError(f"benchmarkingjob' test_object doesn't find"
                             f" the field({test_object_type}).")

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

        self.testcase_controller.build_testcases(test_env=self.test_env,
                                                 test_object=self.test_object)

        succeed_testcases, test_results = self.testcase_controller.run_testcases(self.workspace)

        if test_results:
            self.rank.save(succeed_testcases, test_results, output_dir=self.workspace)
            self.rank.plot()

    def _parse_config(self, config: dict):
        # pylint: disable=C0103
        for k, v in config.items():
            if k == str.lower(TestEnv.__name__):
                self._parse_testenv_config(v)
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
