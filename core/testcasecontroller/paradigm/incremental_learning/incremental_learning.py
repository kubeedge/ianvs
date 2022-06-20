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

import copy
import os

from core.testcasecontroller.paradigm.base import ParadigmBase
from core.testenvmanager.testenv import TestEnv
from core.testcasecontroller.algorithm import Algorithm
from core.testcasecontroller.metrics import get_metric_func
from core.common.utils import get_file_format


class IncrementalLearning(ParadigmBase):
    """ IncrementalLearning pipeline """

    def __init__(self, test_env: TestEnv, algorithm: Algorithm, workspace: str):
        super(IncrementalLearning, self).__init__(test_env, algorithm, workspace)
        self.train_dataset_ratio = algorithm.incremental_learning_data_setting.get("train_ratio")
        self.train_dataset_format = get_file_format(self.dataset.train_url)
        self.splitting_dataset_method = algorithm.incremental_learning_data_setting.get("splitting_method")

    def run(self):
        rounds = self.test_env.incremental_rounds

        try:
            dataset_files = self._preprocess_dataset(splitting_dataset_times=rounds)
        except Exception as err:
            raise Exception(f"preprocess dataset failed, error: {err}.")

        current_model_url = self.algorithm.initial_model_url
        for r in range(1, rounds + 1):
            train_dataset_file, eval_dataset_file = dataset_files[r - 1]

            train_output_dir = os.path.join(self.workspace, f"output/train/{r}")
            os.environ["MODEL_URL"] = train_output_dir
            os.environ["BASE_MODEL_URL"] = current_model_url
            job, feature_process = self.algorithm.build()
            train_dataset = self.load_data(train_dataset_file, "train", feature_process=feature_process)
            new_model_path = job.train(train_dataset)

            os.environ["MODEL_URLS"] = f"{new_model_path};{current_model_url}"
            eval_dataset = self.load_data(eval_dataset_file, "eval", feature_process=feature_process)
            model_eval_info = copy.deepcopy(self.test_env.model_eval)
            model_metric = model_eval_info.get("model_metric")
            metric_name = model_metric.get("name")
            eval_results = job.evaluate(eval_dataset, metric=get_metric_func(model_metric))

            operator_info = model_eval_info
            if self._trigger_deploy(eval_results, metric_name, operator_info):
                current_model_url = new_model_path

            inference_dataset = self.load_data(self.dataset.test_url, "inference", feature_process=feature_process)
            inference_output_dir = os.path.join(self.workspace, f"output/inference/{r}")
            os.environ["INFERENCE_OUTPUT_DIR"] = inference_output_dir
            os.environ["MODEL_URL"] = current_model_url
            infer_res, _, _ = job.inference(inference_dataset.x)

        return self.eval_overall(infer_res)

    def _trigger_deploy(self, eval_results, metric_name, operator_info):
        operator = operator_info.get("operator")
        threshold = operator_info.get("threshold")

        operator_map = {
            ">": lambda x, y: x > y,
            "<": lambda x, y: x < y,
            "=": lambda x, y: x == y,
            ">=": lambda x, y: x >= y,
            "<=": lambda x, y: x <= y,
        }

        if operator not in operator_map:
            raise ValueError(f"operator {operator} use to compare is not allow, set to <")

        operator_func = operator_map[operator]

        if len(eval_results) != 2:
            raise Exception(f"two models of evaluation should have two results. the eval results: {eval_results}")

        metric_values = [0, 0]
        for i, result in enumerate(eval_results):
            metrics = result.get("metrics")
            metric_values[i] = metrics.get(metric_name)

        metric_delta = metric_values[0] - metric_values[1]
        return operator_func(metric_delta, threshold)

    def _preprocess_dataset(self, splitting_dataset_times=1):
        return self.dataset.splitting_dataset(self.dataset.train_url,
                                              self.train_dataset_format,
                                              self.train_dataset_ratio,
                                              method=self.splitting_dataset_method,
                                              output_dir=self.dataset_output_dir(),
                                              times=splitting_dataset_times)
