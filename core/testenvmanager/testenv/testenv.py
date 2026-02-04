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
                "mode": "",
                "name": "",
                "url": "",
            },
            "threshold": 0.9,
            "operator": ">",
        }
        self.metrics = []
        self.incremental_rounds = 2
        self.round = 1
        self.client_number = 1
        self.dataset = None
        self.use_gpu = False  # default false
        self._parse_config(config)

    def _check_fields(self):
        if not self.metrics:
            raise ValueError(f"not found testenv metrics({self.metrics}).")

        if not isinstance(self.incremental_rounds, int) or self.incremental_rounds < 2:
            raise ValueError(
                f"testenv incremental_rounds(value={self.incremental_rounds})"
                f" must be int type and not less than 2."
            )

        if not isinstance(self.round, int) or self.round < 1:
            raise ValueError(f"testenv round(value={self.round}) must be int type and not less than 1.")

        if not isinstance(self.client_number, int) or self.client_number < 1:
            raise ValueError(
                f"testenv client_number(value={self.client_number}) must be int type and not less than 1."
            )

    def _parse_config(self, config):
        config_dict = config.get(str.lower(TestEnv.__name__))
        if not config_dict:
            raise ValueError(f"not found {str.lower(TestEnv.__name__)} in config.")
        fields_mapping = {
            "model_eval": dict,
            "metrics": list,
            "incremental_rounds": int,
            "round": int,
            "client_number": int,
            "use_gpu": bool,
        }

        for k, v in config_dict.items():
            if k == str.lower(Dataset.__name__):
                self.dataset = Dataset(v)
            elif k in fields_mapping:
                expected_type = fields_mapping[k]
                if not isinstance(v, expected_type):
                    try:
                        if expected_type == bool:
                            casted_v = str(v).lower() in ("true", "1", "yes")
                        else:
                            casted_v = expected_type(v)
                        setattr(self, k, casted_v)
                    except (ValueError, TypeError) as err:
                        raise ValueError(
                            f"testenv field '{k}' expected {expected_type.__name__}, "
                            f"got {type(v).__name__} (value={v}). Error: {err}"
                        ) from err
                else:
                    setattr(self, k, v)

        self._check_fields()

    def prepare(self):
        """prepare env"""
        try:
            self.dataset.process_dataset()
        except Exception as err:
            raise RuntimeError(f"prepare dataset failed, error: {err}.") from err
