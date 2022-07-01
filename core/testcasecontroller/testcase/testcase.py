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

"""Test Case"""

import os
import uuid

from core.common.constant import SystemMetricKind
from core.testcasecontroller.metrics import get_metric_func


class TestCase:
    """
    Test Case:
    Consists of a test environment and a test algorithm

    Parameters
    ----------
    test_env : instance
        The test environment of  benchmarking,
        including dataset, Post-processing algorithms like metric computation.
    algorithm : instance
        Typical distributed-synergy AI algorithm paradigm.
    """

    def __init__(self, test_env, algorithm):
        # pylint: disable=C0103
        self.id = uuid.uuid1()
        self.test_env = test_env
        self.algorithm = algorithm
        self.output_dir = None

    def _get_output_dir(self, workspace):
        output_dir = os.path.join(workspace, self.algorithm.name)
        flag = True
        while flag:
            output_dir = os.path.join(workspace, self.algorithm.name, str(self.id))
            if not os.path.exists(output_dir):
                flag = False
        return output_dir

    def run(self, workspace):
        """
        Run the test case

        Returns
        -------
        test result: dict
            e.g.: {"f1_score": 0.89}
        """

        try:
            dataset = self.test_env.dataset
            test_env_config = {}
            # pylint: disable=C0103
            for k, v in self.test_env.__dict__.items():
                test_env_config[k] = v

            self.output_dir = self._get_output_dir(workspace)
            paradigm = self.algorithm.paradigm(workspace=self.output_dir,
                                               **test_env_config)
            res, system_metric_info = paradigm.run()
            test_result = self.compute_metrics(res, dataset, **system_metric_info)

        except Exception as err:
            paradigm_kind = self.algorithm.paradigm_kind
            raise Exception(
                f"(paradigm={paradigm_kind}) pipeline runs failed, error: {err}") from err
        return test_result

    def compute_metrics(self, paradigm_result, dataset, **kwargs):
        """
        Compute metrics of paradigm result

        Parameters
        ----------
        paradigm_result: numpy.ndarray
        dataset: instance
        kwargs: dict
            information needed to compute system metrics.

        Returns
        -------
        dict
            e.g.: {"f1_score": 0.89}
        """

        metric_funcs = {}
        for metric_dict in self.test_env.metrics:
            metric_name, metric_func = get_metric_func(metric_dict=metric_dict)
            if callable(metric_func):
                metric_funcs.update({metric_name: metric_func})

        test_dataset_file = dataset.test_url
        test_dataset = dataset.load_data(test_dataset_file,
                                         data_type="eval overall",
                                         label=dataset.label)

        metric_res = {}
        system_metric_kinds = [e.value for e in SystemMetricKind.__members__.values()]
        for metric_name, metric_func in metric_funcs.items():
            if metric_name in system_metric_kinds:
                metric_res[metric_name] = metric_func(kwargs)
            else:
                metric_res[metric_name] = metric_func(test_dataset.y, paradigm_result)

        return metric_res
