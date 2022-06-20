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

from core.testenvmanager.dataset import Dataset


class TestEnv:
    def __init__(self):
        self.dataset = Dataset()
        self.model_eval = {
            "model_metric": {
                "name": "",
                "parameters": {},
            },
            "threshold": 0.9,
            "operator": ">"
        }
        self.metrics = []
        self.incremental_rounds = 2

    def check_fields(self):
        self.dataset.check_fields()
        if not self.metrics:
            raise ValueError(f"not found testenv metrics({self.metrics}).")

    def prepare(self):
        """ prepare env"""
        try:
            self.dataset.process_dataset()
        except Exception as err:
            raise Exception(f"prepare dataset failed, error: {err}.")
