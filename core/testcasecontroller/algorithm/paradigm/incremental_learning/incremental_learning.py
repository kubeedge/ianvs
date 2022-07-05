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

"""Incremental Learning Paradigm"""

import os
import shutil
import tempfile

import numpy as np

from core.common.constant import ParadigmType, SystemMetricType
from core.testcasecontroller.algorithm.paradigm.base import ParadigmBase
from core.testcasecontroller.metrics import get_metric_func
from core.common.utils import get_file_format, is_local_dir


class IncrementalLearning(ParadigmBase):
    """
    IncrementalLearning

    provide the flow of incremental learning paradigm.

    Parameters
    ---------
    workspace: string
        the output required for single task learning paradigm.
    kwargs: dict
        config required for the test process of incremental learning paradigm,
        e.g.: algorithm modules, dataset, initial model, incremental rounds,
              model eval config, etc.

    """

    def __init__(self, workspace, **kwargs):
        ParadigmBase.__init__(self, workspace, **kwargs)

        self.incremental_learning_data_setting = kwargs.get("incremental_learning_data_setting")
        self.initial_model = kwargs.get("initial_model_url")

        self.incremental_rounds = kwargs.get("incremental_rounds", 2)
        self.model_eval_config = kwargs.get("model_eval")

        self.system_metric_info = {SystemMetricType.SAMPLES_TRANSFER_RATIO.value: []}

    def run(self):
        """
        run the test flow of incremental learning paradigm.

        Returns
        ------
        test result: numpy.ndarray
        system metric info: dict
            information needed to compute system metrics.

        """

        rounds = self.incremental_rounds
        samples_transfer_ratio_info = self.system_metric_info.get(
            SystemMetricType.SAMPLES_TRANSFER_RATIO.value)
        dataset_files = self._preprocess_dataset(splitting_dataset_times=rounds)
        current_model = self.initial_model

        # pylint: disable=C0103
        for r in range(1, rounds + 1):
            inference_dataset_file, eval_dataset_file = dataset_files[r - 1]

            inference_results, hard_examples = self._inference(current_model,
                                                               inference_dataset_file,
                                                               r)

            samples_transfer_ratio_info.append((inference_results, hard_examples))

            # If no hard examples in the first round, starting the next round
            if len(hard_examples) <= 0:
                continue

            train_dataset_file = self._get_train_dataset(hard_examples, inference_dataset_file)

            new_model = self._train(current_model, train_dataset_file, r)

            eval_results = self._eval(new_model, current_model, eval_dataset_file)

            if self._trigger_model_update(eval_results):
                current_model = new_model

        test_res, hard_examples = self._inference(current_model, self.dataset.test_url, "test")
        samples_transfer_ratio_info.append((test_res, hard_examples))

        return test_res, self.system_metric_info

    def _prepare_inference(self, model, rounds):
        inference_output_dir = os.path.join(self.workspace,
                                            f"output/inference/results/{rounds}")
        if not is_local_dir(inference_output_dir):
            os.makedirs(inference_output_dir)

        hard_example_saved_dir = os.path.join(self.workspace,
                                              f"output/inference/hard_examples/{rounds}")
        if not is_local_dir(hard_example_saved_dir):
            os.makedirs(hard_example_saved_dir)

        os.environ["RESULT_SAVED_URL"] = inference_output_dir
        os.environ["MODEL_URL"] = model

        return hard_example_saved_dir

    def _inference(self, model, data_index_file, rounds):
        hard_example_saved_dir = self._prepare_inference(model, rounds)

        job = self.build_paradigm_job(ParadigmType.INCREMENTAL_LEARNING.value)
        inference_dataset = self.dataset.load_data(data_index_file, "inference")
        inference_dataset_x = inference_dataset.x

        inference_results = {}
        hard_examples = []
        for _, data in enumerate(inference_dataset_x):
            res, _, is_hard_example = job.inference([data])
            inference_results.update(res)
            if is_hard_example:
                shutil.copy(data, hard_example_saved_dir)
                new_hard_example = os.path.join(hard_example_saved_dir, os.path.basename(data))
                hard_examples.append((data, new_hard_example))
        del job

        return inference_results, hard_examples

    def _get_train_dataset(self, hard_examples, data_label_file):
        data_labels = self.dataset.load_data(data_label_file, "train label")
        temp_dir = tempfile.mkdtemp()
        train_dataset_file = os.path.join(temp_dir, os.path.basename(data_label_file))
        with open(train_dataset_file, "w", encoding="utf-8") as file:
            for old, new in hard_examples:
                index = np.where(data_labels.x == old)
                if len(index[0]) == 1:
                    label = data_labels.y[index[0][0]]
                file.write(f"{new} {label}\n")

        return train_dataset_file

    def _train(self, model, data_index_file, rounds):
        train_output_dir = os.path.join(self.workspace, f"output/train/{rounds}")
        if not is_local_dir(train_output_dir):
            os.makedirs(train_output_dir)

        os.environ["MODEL_URL"] = train_output_dir
        os.environ["BASE_MODEL_URL"] = model

        job = self.build_paradigm_job(ParadigmType.INCREMENTAL_LEARNING.value)
        train_dataset = self.dataset.load_data(data_index_file, "train")
        new_model = job.train(train_dataset)
        del job

        return new_model

    def _eval(self, new_model, old_model, data_index_file):
        os.environ["MODEL_URLS"] = f"{new_model};{old_model}"
        model_eval_info = self.model_eval_config
        model_metric = model_eval_info.get("model_metric")

        job = self.build_paradigm_job(ParadigmType.INCREMENTAL_LEARNING.value)
        eval_dataset = self.dataset.load_data(data_index_file, "eval")
        eval_results = job.evaluate(eval_dataset, metric=get_metric_func(model_metric))
        del job

        return eval_results

    def _trigger_model_update(self, eval_results):
        model_eval_info = self.model_eval_config
        model_metric = model_eval_info.get("model_metric")
        metric_name = model_metric.get("name")
        operator = model_eval_info.get("operator")
        threshold = model_eval_info.get("threshold")

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
            raise Exception(f"two models of evaluation should have two results."
                            f" the eval results: {eval_results}")

        metric_values = [0, 0]
        for i, result in enumerate(eval_results):
            metrics = result.get("metrics")
            metric_values[i] = metrics.get(metric_name)

        metric_delta = metric_values[0] - metric_values[1]
        return operator_func(metric_delta, threshold)

    def _preprocess_dataset(self, splitting_dataset_times=1):
        train_dataset_ratio = self.incremental_learning_data_setting.get("train_ratio")
        splitting_dataset_method = self.incremental_learning_data_setting.get("splitting_method")

        return self.dataset.split_dataset(self.dataset.train_url,
                                          get_file_format(self.dataset.train_url),
                                          train_dataset_ratio,
                                          method=splitting_dataset_method,
                                          dataset_types=("model_inference", "model_evaluation"),
                                          output_dir=self.dataset_output_dir(),
                                          times=splitting_dataset_times)
