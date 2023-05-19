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

"""Lifelong Learning Paradigm"""

import os
import shutil
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sedna.datasources import BaseDataSource

from core.common.constant import ParadigmType, SystemMetricType
from core.testcasecontroller.algorithm.paradigm.base import ParadigmBase
from core.testcasecontroller.metrics import get_metric_func
from core.common.utils import get_file_format, is_local_dir

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class LifelongLearning(ParadigmBase):
    # pylint: disable=too-many-locals
    """
    LifelongLearning
    provide the flow of lifelong learning paradigm.
    Notes:
          1. Ianvs serves as testing tools for test objects, e.g., algorithms.
          2. Ianvs does NOT include code directly on test object.
          3. Algorithms serve as typical test objects in Ianvs
          and detailed algorithms are thus NOT included in this Ianvs python file.
          4. As for the details of example test objects, e.g., algorithms,
          please refer to third party packages in Ianvs example.
          For example, AI workflow and interface pls refer to sedna
          (sedna docs: https://sedna.readthedocs.io/en/latest/api/lib/index.html),
          and module implementation pls refer to `examples' test algorithms`,
          e.g., basemodel.py, hard_example_mining.py.

    Parameters
    ---------
    workspace: string
        the output required for lifelong learning paradigm.
    kwargs: dict
        config required for the test process of lifelong learning paradigm,
        e.g.: algorithm modules, dataset, initial model, incremental rounds,
              model eval config, etc.

    """

    def __init__(self, workspace, **kwargs):
        ParadigmBase.__init__(self, workspace, **kwargs)

        self.incremental_learning_data_setting = kwargs.get("lifelong_learning_data_setting")
        self.initial_model = kwargs.get("initial_model_url")
        self.incremental_rounds = kwargs.get("incremental_rounds", 1)
        self.model_eval_config = kwargs.get("model_eval")
        self.cloud_task_index = '/tmp/cloud_task/index.pkl'
        self.edge_task_index = '/tmp/edge_task/index.pkl'
        self.system_metric_info = {SystemMetricType.SAMPLES_TRANSFER_RATIO.value: [],
                                   SystemMetricType.BWT.value : {}, SystemMetricType.FWT.value: {}}

    def run(self):
        # pylint:disable=duplicate-code
        # pylint: disable=R0912
        # pylint: disable=R0915
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
        mode = self.model_eval_config.get("model_metric").get("mode")

        # in this mode, the inference period is skipped to accelerate training speed
        if mode == 'no-inference':
            dataset_files = self._split_dataset(splitting_dataset_times=rounds)
            # pylint: disable=C0103
            # pylint: disable=C0206
            # pylint: disable=C0201
            my_dict = {}
            for r in range(1, rounds + 1):
                if r == 1:
                    train_dataset_file, eval_dataset_file = dataset_files[r - 1]
                    self.cloud_task_index = self._train(self.cloud_task_index,
                                                        train_dataset_file,
                                                        r)
                    tmp_dict = {}
                    for j in range(rounds):
                        _, eval_dataset_file = dataset_files[j]
                        self.edge_task_index, tasks_detail, res = self.my_eval(
                                                    self.cloud_task_index,
                                                      eval_dataset_file,
                                                      r)
                        print("train from round ", r)
                        print("test round ", j+1)
                        print(f"all scores: {res}")
                        score_list = tmp_dict.get("all", [])
                        score_list.append(res)
                        tmp_dict["all"] = score_list
                        for detail in tasks_detail:
                            scores = detail.scores
                            entry = detail.entry
                            print(f"{entry} scores: {scores}")
                            score_list = tmp_dict.get(entry, [])
                            score_list.append(scores)
                            tmp_dict[entry] = score_list
                    for key in tmp_dict.keys():
                        scores_list = my_dict.get(key, [])
                        scores_list.append(tmp_dict[key])
                        my_dict[key] = scores_list
                        print(f"{key} scores: {scores_list}")

                else:
                    infer_dataset_file, eval_dataset_file = dataset_files[r - 1]
                    self.cloud_task_index = self._train(self.cloud_task_index,
                                                        infer_dataset_file,
                                                        r)
                    tmp_dict = {}
                    for j in range(rounds):
                        _, eval_dataset_file = dataset_files[j]
                        self.edge_task_index, tasks_detail, res = self.my_eval(
                                                    self.cloud_task_index,
                                                      eval_dataset_file,
                                                      r)
                        print("train from round ", r)
                        print("test round ", j+1)
                        print(f"all scores: {res}")
                        score_list = tmp_dict.get("all", [])
                        score_list.append(res)
                        tmp_dict["all"] = score_list
                        for detail in tasks_detail:
                            scores = detail.scores
                            entry = detail.entry
                            print(f"{entry} scores: {scores}")
                            score_list = tmp_dict.get(entry, [])
                            score_list.append(scores)
                            tmp_dict[entry] = score_list
                    for key in tmp_dict.keys():
                        scores_list = my_dict.get(key, [])
                        scores_list.append(tmp_dict[key])
                        my_dict[key] = scores_list
                        print(f"{key} scores: {scores_list}")

            self.edge_task_index = self._eval(self.cloud_task_index,
                                                      self.dataset.test_url,
                                                      r+1)
            job = self.build_paradigm_job(ParadigmType.LIFELONG_LEARNING.value)
            inference_dataset = self.dataset.load_data(self.dataset.test_url, "eval",
                                                   feature_process=_data_feature_process)
            kwargs = {}
            test_res = job.my_inference(inference_dataset, **kwargs)
            del job
            output_dir = os.path.join(self.workspace,
                                  "output/result.pkl")
            for key in my_dict.keys():
                print(f"{key} scores: {my_dict[key]}")
            with open(output_dir,'wb') as file:
                pickle.dump(my_dict,file)
            for key in my_dict.keys():
                matrix = my_dict[key]
                BWT, FWT = self.compute(key, matrix)
                self.system_metric_info[SystemMetricType.BWT.value][key] = BWT
                self.system_metric_info[SystemMetricType.FWT.value][key] = FWT

            file.close()

        elif mode != 'multi-inference':
            dataset_files = self._split_dataset(splitting_dataset_times=rounds)
            # pylint: disable=C0103
            for r in range(1, rounds + 1):
                if r == 1:
                    train_dataset_file, eval_dataset_file = dataset_files[r - 1]
                    self.cloud_task_index = self._train(self.cloud_task_index,
                                                        train_dataset_file,
                                                        r)
                    self.edge_task_index = self._eval(self.cloud_task_index,
                                                      eval_dataset_file,
                                                      r)
                else:
                    infer_dataset_file, eval_dataset_file = dataset_files[r - 1]

                    inference_results, unseen_task_train_samples = self._inference(
                                                    self.edge_task_index,
                                                    infer_dataset_file,
                                                    r)
                    samples_transfer_ratio_info.append((inference_results,
                                                unseen_task_train_samples.x))

                    # If no unseen task samples in the this round, starting the next round
                    if len(unseen_task_train_samples.x) <= 0:
                        continue

                    self.cloud_task_index = self._train(self.cloud_task_index,
                                                        unseen_task_train_samples,
                                                        r)
                    self.edge_task_index = self._eval(self.cloud_task_index,
                                                      eval_dataset_file,
                                                      r)
            test_res, unseen_task_train_samples = self._inference(self.edge_task_index,
                                                              self.dataset.test_url,
                                                              "test")

        return test_res, self.system_metric_info

    def compute(self, key, matrix):
        """
        compute BWT and FWT
        """
        # pylint: disable=C0103
        length = len(matrix)
        accuracy = 0.0
        BWT_score = 0.0
        FWT_score = 0.0
        flag = True
        for i in range(length):
            if len(matrix[i]) != length:
                flag = False
                break
        if flag is False:
            BWT_score = np.nan
            FWT_score = np.nan
            return BWT_score, FWT_score

        for i in range(length):
            accuracy += matrix[length-1][i]['accuracy']
            BWT_score += matrix[length-1][i]['accuracy'] - matrix[i][i]['accuracy']
        for i in range(1,length):
            FWT_score += matrix[i-1][i]['accuracy'] - matrix[0][i]['accuracy']
        accuracy = accuracy/(length)
        BWT_score = BWT_score/(length-1)
        FWT_score = FWT_score/(length-1)
        #print(f"{key} accuracy: ", accuracy)
        print(f"{key} BWT_score: ", BWT_score)
        print(f"{key} FWT_score: ", FWT_score)
        my_matrix = []
        for i in range(length):
            my_matrix.append([])
        for i in range(length):
            for j in range(length):
                my_matrix[i].append(matrix[j][i]['accuracy'])
        self.draw_picture(key,my_matrix)
        return BWT_score, FWT_score

    def draw_picture(self, title, matrix):
        """
        draw hot image for results
        """
        plt.figure(figsize=(10,8), dpi=80)
        plt.imshow(matrix, cmap='bwr', extent=(0.5,len(matrix)+0.5,0.5,len(matrix)+0.5),
                    origin='lower')
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel('task round', fontsize=15)
        plt.ylabel('task', fontsize=15)
        plt.title(title, fontsize=15)
        plt.colorbar(format='%.2f')
        output_dir = os.path.join(self.workspace,
                                  f"output/{title}-heatmap.png")
        plt.savefig(output_dir)
        plt.show()

    def _inference(self, edge_task_index, data_index_file, rounds):
        # pylint:disable=duplicate-code
        #print("start inference")
        output_dir = os.path.join(self.workspace,
                                  f"output/inference/results/{rounds}")
        if not is_local_dir(output_dir):
            os.makedirs(output_dir)

        unseen_task_saved_dir = os.path.join(self.workspace,
                                        f"output/inference/unseen_task_samples/{rounds}")
        if not is_local_dir(unseen_task_saved_dir):
            os.makedirs(unseen_task_saved_dir)

        os.environ["INFERENCE_RESULT_DIR"] = output_dir
        os.environ["MODEL_URLS"] = f"{edge_task_index}"

        inference_dataset = self.dataset.load_data(data_index_file, "eval",
                                                   feature_process=_data_feature_process)

        job = self.build_paradigm_job(ParadigmType.LIFELONG_LEARNING.value)

        inference_results = []
        unseen_tasks = []
        unseen_task_labels = []
        mode = self.model_eval_config.get("model_metric").get("mode")
        if mode is None:
            kwargs = {}
            # fix the bug of "TypeError: call() got an unexpected keyword argument 'mode'"
        else:
            kwargs = {"mode": mode}
        #print(len(inference_dataset.x))
        for i, _ in enumerate(inference_dataset.x):
            data = BaseDataSource(data_type="test")
            data.x = inference_dataset.x[i:(i + 1)]
            res, is_unseen_task, _ = job.inference(data, **kwargs)
            inference_results.append(res)
            if is_unseen_task:
                unseen_tasks.append(inference_dataset.x[i])
                unseen_task_labels.append(inference_dataset.y[i])
                for infer_data in inference_dataset.x[i]:
                    shutil.copy(infer_data, unseen_task_saved_dir)

        del job

        unseen_task_train_samples = BaseDataSource(data_type="train")
        unseen_task_train_samples.x = np.array(unseen_tasks)
        unseen_task_train_samples.y = np.array(unseen_task_labels)

        return inference_results, unseen_task_train_samples

    def _train(self, cloud_task_index, train_dataset, rounds):
        train_output_dir = os.path.join(self.workspace, f"output/train/{rounds}")
        if not is_local_dir(train_output_dir):
            os.makedirs(train_output_dir)

        os.environ["CLOUD_KB_INDEX"] = cloud_task_index
        os.environ["OUTPUT_URL"] = train_output_dir
        if rounds == 1:
            os.environ["HAS_COMPLETED_INITIAL_TRAINING"] = 'False'
        else:
            os.environ["HAS_COMPLETED_INITIAL_TRAINING"] = 'True'

        if isinstance(train_dataset, str):
            train_dataset = self.dataset.load_data(train_dataset, "train",
                                                   feature_process=_data_feature_process)

        job = self.build_paradigm_job(ParadigmType.LIFELONG_LEARNING.value)
        cloud_task_index = job.train(train_dataset)
        del job

        return cloud_task_index

    def _eval(self, cloud_task_index, data_index_file, rounds):
        eval_output_dir = os.path.join(self.workspace, f"output/eval/{rounds}")
        if not is_local_dir(eval_output_dir):
            os.makedirs(eval_output_dir)

        model_eval_info = self.model_eval_config
        model_metric = model_eval_info.get("model_metric")

        os.environ["OUTPUT_URL"] = eval_output_dir
        os.environ["model_threshold"] = str(model_eval_info.get("threshold"))
        os.environ["operator"] = model_eval_info.get("operator")
        os.environ["MODEL_URLS"] = f"{cloud_task_index}"

        eval_dataset = self.dataset.load_data(data_index_file, "eval",
                                              feature_process=_data_feature_process)

        job = self.build_paradigm_job(ParadigmType.LIFELONG_LEARNING.value)
        _, metric_func = get_metric_func(model_metric)
        edge_task_index = job.evaluate(eval_dataset, metrics=metric_func)

        del job

        return edge_task_index

    def my_eval(self, cloud_task_index, data_index_file, rounds):
        """
        evaluate models
        """
        eval_output_dir = os.path.join(self.workspace, f"output/eval/{rounds}")
        if not is_local_dir(eval_output_dir):
            os.makedirs(eval_output_dir)

        model_eval_info = self.model_eval_config
        model_metric = model_eval_info.get("model_metric")

        os.environ["OUTPUT_URL"] = eval_output_dir
        os.environ["model_threshold"] = str(model_eval_info.get("threshold"))
        os.environ["operator"] = model_eval_info.get("operator")
        os.environ["MODEL_URLS"] = f"{cloud_task_index}"

        eval_dataset = self.dataset.load_data(data_index_file, "eval",
                                              feature_process=_data_feature_process)

        job = self.build_paradigm_job(ParadigmType.LIFELONG_LEARNING.value)
        _, metric_func = get_metric_func(model_metric)
        edge_task_index, tasks_detail, res = job.my_evaluate(eval_dataset, metrics=metric_func)

        del job

        return edge_task_index, tasks_detail, res

    def _split_dataset(self, splitting_dataset_times=1):
        # pylint:disable=duplicate-code
        train_dataset_ratio = self.incremental_learning_data_setting.get("train_ratio")
        splitting_dataset_method = self.incremental_learning_data_setting.get("splitting_method")

        return self.dataset.split_dataset(self.dataset.train_url,
                                          get_file_format(self.dataset.train_url),
                                          train_dataset_ratio,
                                          method=splitting_dataset_method,
                                          dataset_types=("model_train", "model_eval"),
                                          output_dir=self.dataset_output_dir(),
                                          times=splitting_dataset_times)


def _data_feature_process(line: str):
    res = line.strip().split()
    return res[:-1], res[-1]
