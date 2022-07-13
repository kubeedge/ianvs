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

"""Test Env"""

from core.testenvmanager.dataset import Dataset

# pylint: disable=too-few-public-methods
class TestEnv:
    """
    TestEnv:
        the test environment of  benchmarking,
        including dataset, Post-processing algorithms like metric computation.

    Parameters
    ----------
    config: dict
        config of the test environment of  benchmarking, includes: dataset, metrics, etc.

    """

    def __init__(self, config):
        self.model_eval = {
            "model_metric": {
                "name": "",
                "url": "",
            },
            "threshold": 0.9,
            "operator": ">"
        }
        self.metrics = []
        self.incremental_rounds = 2
        self.dataset = None
        self._parse_config(config)

    def _check_fields(self):
        if not self.metrics:
            raise ValueError(f"not found testenv metrics({self.metrics}).")

    def _parse_config(self, config):
        config_dict = config[str.lower(TestEnv.__name__)]
        # pylint: disable=C0103
        for k, v in config_dict.items():
            if k == str.lower(Dataset.__name__):
                self.dataset = Dataset(v)
            else:
                if k in self.__dict__:
                    self.__dict__[k] = v

        self._check_fields()

    def prepare(self):
        """ prepare env"""
        try:
            self.dataset.process_dataset()
        except Exception as err:
            raise Exception(f"prepare dataset failed, error: {err}.") from err
