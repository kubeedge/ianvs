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

import os
import uuid

from core.testcasecontroller.paradigm import Paradigm


class TestCase:
    def __init__(self, test_env, algorithm):
        """
        Distributed collaborative AI algorithm in certain test environment
        Parameters
        ----------
        test_env : instance
            The test environment of  distributed collaborative AI benchmark
            including samples, dataset setting, metrics
        algorithm : instance
            Distributed collaborative AI algorithm
        """
        self.test_env = test_env
        self.algorithm = algorithm

    def prepare(self, metrics, workspace):
        self.id = self._get_id()
        self.output_dir = self._get_output_dir(workspace)
        self.metrics = metrics

    def _get_output_dir(self, workspace):
        output_dir = os.path.join(workspace, self.algorithm.name)
        flag = True
        while flag:
            output_dir = os.path.join(workspace, self.algorithm.name, str(self.id))
            if not os.path.exists(output_dir):
                flag = False
        return output_dir

    def _get_id(self):
        return uuid.uuid1()

    def run(self):
        try:
            paradigm = Paradigm(self.algorithm.paradigm, self.test_env, self.algorithm, self.output_dir)
            res = paradigm.run()
        except Exception as err:
            raise Exception(f"(paradigm={self.algorithm.paradigm}) pipeline runs failed, error: {err}")
        return res
