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

"""Federated Class-Incremental Learning Paradigm"""
# pylint: disable=C0412
# pylint: disable=W1203
# pylint: disable=C0103
# pylint: disable=C0206
# pylint: disable=C0201
import numpy as np
from core.common.log import LOGGER
from core.common.constant import ParadigmType, SystemMetricType
from core.testcasecontroller.metrics.metrics import get_metric_func
from .federated_learning import FederatedLearning


class FederatedClassIncrementalLearning(FederatedLearning):
    # pylint: disable=too-many-instance-attributes
    """
    FederatedClassIncrementalLearning
    Federated Class-Incremental Learning Paradigm
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
        the output required for Federated Class-Incremental Learning paradigm.
    kwargs: dict
        config required for the test process of lifelong learning paradigm,
        e.g.: algorithm modules, dataset, initial network, incremental rounds,
              network eval config, etc.
    """

    def __init__(self, workspace, **kwargs):
        super().__init__(workspace, **kwargs)
        self.incremental_rounds = kwargs.get("incremental_rounds", 1)
        self.system_metric_info = {
            SystemMetricType.FORGET_RATE.value: [],
            SystemMetricType.TASK_AVG_ACC.value: {},
        }

        self.aggregate_clients = []
        self.train_infos = []

        self.forget_rate_metrics = []
        self.accuracy_per_round = []
        metrics_dict = kwargs.get("model_eval", {})["model_metric"]
        _, accuracy_func = get_metric_func(metrics_dict)
        self.accuracy_func = accuracy_func

    def task_definition(self, dataset_files, task_id):
        """Define the task for the class incremental learning paradigm

        Args:
            dataset_files (list): dataset_files for train data
            task_id (int): task id for the current task

        Returns:
            list: train dataset in numpy format for each task
        """
        LOGGER.info(f"len(dataset_files): {len(dataset_files)}")
        # 1. Partition Dataset
        train_dataset_files, _ = dataset_files[task_id]
        LOGGER.info(f"train_dataset_files: {train_dataset_files}")
        train_datasets = self.train_data_partition(train_dataset_files)
        LOGGER.info(f"train_datasets: {len(train_datasets)}")
        task_size = self.get_task_size(train_datasets)
        LOGGER.info(f"task_size: {task_size}")
        # 2. According to setting, to split the label and unlabel data for each task
        train_datasets = self.split_label_unlabel_data(train_datasets)
        # 3. Return the dataset for each task [{label_data, unlabel_data}, ...]
        return train_datasets, task_size

    def get_task_size(self, train_datasets):
        """get the task size for each task

        Args:
            train_datasets (list): train dataset for each client

        Returns:
            int: task size for each task
        """
        LOGGER.info(f"train_datasets: {len(train_datasets[0])}")
        return np.unique(train_datasets[0][1]).shape[0]

    def split_label_unlabel_data(self, train_datasets):
        """split train dataset into label and unlabel data for semi-supervised learning

        Args:
            train_datasets (list): train dataset for each client

        Returns:
            list: the new train dataset for each client that in label and unlabel format
            [{label_x: [], label_y: [], unlabel_x: [], unlabel_y: []}, ...]
        """
        label_ratio = self.fl_data_setting.get("label_data_ratio")
        new_train_datasets = []
        train_dataset_len = len(train_datasets)
        for i in range(train_dataset_len):
            train_dataset_dict = {}
            label_data_number = int(label_ratio * len(train_datasets[i][0]))
            # split dataset into label and unlabel data
            train_dataset_dict["label_x"] = train_datasets[i][0][:label_data_number]
            train_dataset_dict["label_y"] = train_datasets[i][1][:label_data_number]
            train_dataset_dict["unlabel_x"] = train_datasets[i][0][label_data_number:]
            train_dataset_dict["unlabel_y"] = train_datasets[i][1][label_data_number:]
            new_train_datasets.append(train_dataset_dict)
        return new_train_datasets

    def init_client(self):
        self.clients = [
            self.build_paradigm_job(
                ParadigmType.FEDERATED_CLASS_INCREMENTAL_LEARNING.value
            )
            for _ in range(self.clients_number)
        ]

    def run(self):
        """run the Federated Class-Incremental Learning paradigm
            This function will run the Federated Class-Incremental Learning paradigm.
            1. initialize the clients
            2. split the dataset into several tasks
            3. train the model on the clients
            4. aggregate the model weights and maybe need to perform some helper function
            5. send the weights to the clients
            6. evaluate the model performance on old classes
            7. finally, return the prediction result and system metric information
        Returns:
            list: prediction result
            dict: system metric information
        """

        self.init_client()
        dataset_files = self._split_dataset(self.incremental_rounds)
        test_dataset_files = self._split_test_dataset(self.incremental_rounds)
        LOGGER.info(f"get the dataset_files: {dataset_files}")
        forget_rate = self.system_metric_info.get(SystemMetricType.FORGET_RATE.value)
        for task_id in range(self.incremental_rounds):
            train_datasets, task_size = self.task_definition(dataset_files, task_id)
            testdatasets = test_dataset_files[: task_id + 1]
            for r in range(self.rounds):
                LOGGER.info(
                    f"Round {r} task id: {task_id} len of train_datasets: {len(train_datasets)}"
                )
                self.train(
                    train_datasets, task_id=task_id, round=r, task_size=task_size
                )
                global_weights = self.aggregator.aggregate(self.aggregate_clients)
                if hasattr(self.aggregator, "helper_function"):
                    self.helper_function(self.train_infos)
                self.send_weights_to_clients(global_weights)
                self.aggregate_clients.clear()
                self.train_infos.clear()
            forget_rate.append(self.evaluation(testdatasets, task_id))
        test_res = self.predict(self.dataset.test_url)
        return test_res, self.system_metric_info

    def _split_test_dataset(self, split_time):
        """split test dataset
            This function will split a test dataset from test_url into several parts.
            Each part will be used for the evaluation of the model after each round.
        Args:
            split_time (int): the number of split time

        Returns:
            list : the test dataset for each round [{x: [], y: []}, ...]
        """
        test_dataset = self.dataset.load_data(self.dataset.test_url, "eval")
        all_data = len(test_dataset.x)
        step = all_data // split_time
        test_datasets_files = []
        index = 1
        while index <= split_time:
            new_dataset = {}
            if index == split_time:
                new_dataset["x"] = test_dataset.x[step * (index - 1) :]
                new_dataset["y"] = test_dataset.y[step * (index - 1) :]
            else:
                new_dataset["x"] = test_dataset.x[step * (index - 1) : step * index]
                new_dataset["y"] = test_dataset.y[step * (index - 1) : step * index]
            test_datasets_files.append(new_dataset)
            index += 1
        return test_datasets_files

    def client_train(self, client_idx, train_datasets, validation_datasets, **kwargs):
        """client train function that will be called by the thread

        Args:
            client_idx (int): client index
            train_datasets (list): train dataset for each client
            validation_datasets (list): validation dataset for each client
        """
        train_info = super().client_train(
            client_idx, train_datasets, validation_datasets, **kwargs
        )
        with self.lock:
            self.train_infos.append(train_info)

    def helper_function(self, train_infos):
        """helper function for FCI Method
           Many of the FCI algorithms need server to perform some operations
           after the training of each round e.g data generation, model update etc.
        Args:
            train_infos (list of dict): the train info that the clients want to send to the server
        """

        for i in range(self.clients_number):
            helper_info = self.aggregator.helper_function(train_infos[i])
            self.clients[i].helper_function(helper_info)
        LOGGER.info("finish helper function")

    # pylint: disable=too-many-locals
    # pylint: disable=C0200
    def evaluation(self, testdataset_files, incremental_round):
        """evaluate the model performance on old classes

        Args:
            testdataset_files (list): the test dataset for each round
            incremental_round (int): the total incremental training round

        Returns:
            float:  forget rate for the current round
                    reference: https://ieeexplore.ieee.org/document/10574196/
        """
        if self.accuracy_func is None:
            raise ValueError("accuracy function is not defined")
        LOGGER.info("********start evaluation********")
        if isinstance(testdataset_files, str):
            testdataset_files = [testdataset_files]
        job = self.get_global_model()
        # caculate the seen class accuracy
        old_class_acc_list = (
            []
        )  # for current round [class_0: acc_0, class_1: acc1, ....]
        for index in range(len(testdataset_files)):
            acc_list = []
            for data_index in range(len(testdataset_files[index]["x"])):
                data = testdataset_files[index]["x"][data_index]
                res = job.inference([data])
                LOGGER.info(
                    f"label is {testdataset_files[index]['y'][data_index]}, res is {res}"
                )
                acc = self.accuracy_func(
                    [testdataset_files[index]["y"][data_index]], res
                )
                acc_list.append(acc)
            old_class_acc_list.extend(acc_list)
        current_forget_rate = 0.0
        max_acc_sum = 0
        self.accuracy_per_round.append(old_class_acc_list)
        self.system_metric_info[SystemMetricType.TASK_AVG_ACC.value]["accuracy"] = (
            np.mean(old_class_acc_list)
        )
        # caculate the forget rate
        for i in range(len(old_class_acc_list)):
            max_acc_diff = 0
            for j in range(incremental_round):
                acc_per_round = self.accuracy_per_round[j]
                if i < len(acc_per_round):
                    max_acc_diff = max(
                        max_acc_diff, acc_per_round[i] - old_class_acc_list[i]
                    )
            max_acc_sum += max_acc_diff
        current_forget_rate = (
            max_acc_sum / len(old_class_acc_list) if incremental_round > 0 else 0.0
        )
        tavk_avg_acc = self.system_metric_info[SystemMetricType.TASK_AVG_ACC.value][
            "accuracy"
        ]
        LOGGER.info(
            f"for current round: {incremental_round} forget rate: {current_forget_rate}"
            f"task avg acc: {tavk_avg_acc}"
        )
        return current_forget_rate
