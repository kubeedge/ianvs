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

from sedna.datasources import CSVDataParse, TxtDataParse

from core.common.constant import DatasetFormat
from core.common import utils
from core.testcasecontroller.metrics import get_metric_func


class ParadigmBase:
    def __init__(self, test_env, algorithm, workspace):
        self.test_env = test_env
        self.dataset = test_env.dataset
        self.algorithm = algorithm
        self.workspace = workspace
        os.environ["LOCAL_TEST"] = "TRUE"

    def eval_overall(self, result):
        """ eval overall results """
        metric_funcs = []
        for metric_dict in self.test_env.metrics:
            metric = get_metric_func(metric_dict=metric_dict)
            if callable(metric):
                metric_funcs.append(metric)

        test_dataset_file = self.dataset.test_url
        test_dataset = self.load_data(test_dataset_file, data_type="eval overall", label=self.dataset.label)
        metric_res = {}
        for metric in metric_funcs:
            metric_res[metric.__name__] = metric(test_dataset.y, result)
        return metric_res

    def dataset_output_dir(self):
        output_dir = os.path.join(self.workspace, "dataset")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return output_dir

    def load_data(self, file: str, data_type: str, label=None, use_raw=False, feature_process=None):
        format = utils.get_file_format(file)

        if format == DatasetFormat.CSV.value:
            data = CSVDataParse(data_type=data_type, func=feature_process)
            data.parse(file, label=label)
        elif format == DatasetFormat.TXT.value:
            data = TxtDataParse(data_type=data_type, func=feature_process)
            data.parse(file, use_raw=use_raw)

        return data
