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

"""Federated  Learning Paradigm"""
# pylint: disable=C0412
# pylint: disable=W1203
# pylint: disable=C0103
# pylint: disable=C0206
# pylint: disable=C0201
# pylint: disable=W1203
from queue import Queue, Empty
from threading import Thread, RLock

from sedna.algorithms.aggregation import AggClient
from core.common.log import LOGGER
from core.common.constant import ParadigmType, ModuleType
from core.common.utils import get_file_format
from core.testcasecontroller.algorithm.paradigm.base import ParadigmBase
from core.testenvmanager.dataset.utils import (
    partition_llm_data,
    read_data_from_file_to_npy,
    partition_data,
    rename_keys_jsonl,
)


class FederatedLearning(ParadigmBase):
    # pylint: disable=too-many-instance-attributes
    """
    FederatedLearning
    Federated Learning Paradigm
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
        ParadigmBase.__init__(self, workspace, **kwargs)

        self.workspace = workspace
        self.kwargs = kwargs
        LOGGER.info(f"kwargs: {self.kwargs}")
        self.fl_data_setting = kwargs.get("fl_data_setting")
        self.rounds = kwargs.get("round", 1)
        self.clients = []
        self.lock = RLock()
        self.aggregate_clients = []
        self.clients_number = kwargs.get("client_number", 1)
        self.mode = kwargs.get("if_mode_llm", False)
        _, self.aggregator = self.module_instances.get(ModuleType.AGGREGATION.value)
        if self.mode:
            LOGGER.info("Using LLM multi GPU mode for Federated Learning")
            self.gpu_num = kwargs.get("gpu_num", 1)
        LOGGER.info(
            f"rounds: {self.rounds}, clients_number: {self.clients_number}, "
            f"fl_data_setting: {self.fl_data_setting}"
        )

    def init_client(self):
        """init clients for the paradigm of federated learning."""
        self.clients = [
            self.build_paradigm_job(ParadigmType.FEDERATED_LEARNING.value)
            for i in range(self.clients_number)
        ]

    def run(self):
        """
        run the test flow of incremental learning paradigm.

        Returns
        ------
        test result: numpy.ndarray
        system metric info: dict
            information needed to compute system metrics.
        """
        # init client wait for connection
        self.init_client()
        dataset_files = self.get_all_train_data()
        train_dataset_file, _ = dataset_files[0]
        if self.mode:
            rename_keys_jsonl(train_dataset_file)
            train_datasets = self.train_llm_data_partition(train_dataset_file)
            train_fn = self.llm_train
        else:
            train_datasets = self.train_data_partition(train_dataset_file)
            train_fn = self.train
        for r in range(self.rounds):
            train_fn(train_datasets, round=r)
            global_weights = self.aggregator.aggregate(self.aggregate_clients)
            self.send_weights_to_clients(global_weights)
            self.aggregate_clients.clear()
        test_res = self.predict(self.dataset.test_url)
        return test_res, self.system_metric_info

    def get_all_train_data(self):
        """Get all train data for the paradigm of federated learning.

        Returns:
            list: train data list
        """
        split_time = 1  # only one split ——all the data
        return self._split_dataset(split_time)

    def _split_dataset(self, splitting_dataset_times=1):
        """spit the dataset using ianvs dataset.split dataset method

        Args:
            splitting_dataset_times (int, optional): . Defaults to 1.

        Returns:
            list: dataset files
        """
        train_dataset_ratio = self.fl_data_setting.get("train_ratio")
        splitting_dataset_method = self.fl_data_setting.get("splitting_method")
        return self.dataset.split_dataset(
            self.dataset.train_url,
            get_file_format(self.dataset.train_url),
            train_dataset_ratio,
            method=splitting_dataset_method,
            dataset_types=("model_train", "model_eval"),
            output_dir=self.dataset_output_dir(),
            times=splitting_dataset_times,
        )

    def train_data_partition(self, train_dataset_file):
        """
        Partition the dataset for the class incremental learning paradigm
        - i.i.d
        - non-i.i.d
        """
        LOGGER.info("train_dataset_file: %s", train_dataset_file)
        train_datasets = None
        if isinstance(train_dataset_file, str):
            train_datasets = self.dataset.load_data(train_dataset_file, "train")
        if isinstance(train_dataset_file, list):
            train_datasets = []
            for file in train_dataset_file:
                train_datasets.append(self.dataset.load_data(file, "train"))
        assert train_datasets is not None, "train_dataset is None"
        # translate file to real data that can be used in train
        # - provide a default method to read data from file to npy
        # - can support customized method to read data from file to npy
        train_datasets = read_data_from_file_to_npy(train_datasets)
        # Partition data to iid or non-iid
        train_datasets = partition_data(
            train_datasets,
            self.clients_number,
            self.fl_data_setting.get("data_partition"),
            self.fl_data_setting.get("non_iid_ratio"),
        )
        return train_datasets

    def train_llm_data_partition(self, train_dataset_file):
        """
        Partition the dataset for llm federated learning paradigm
        
        Returns:
            list: train data for each client
        """
        LOGGER.info("train_dataset_file: %s", train_dataset_file)
        train_datasets = None
        if isinstance(train_dataset_file, str):
            if get_file_format(train_dataset_file) == "jsonl":
                train_datasets = self.dataset.load_data(train_dataset_file, data_type="train")
            else:
                raise ValueError(
                    f"LLM Unsupported file format: {get_file_format(train_dataset_file)}. "
                    f"Only jsonl is supported."
                )
        assert train_datasets is not None, "train_dataset is None"
        train_datasets = partition_llm_data(train_datasets, self.clients_number)
        return train_datasets

    def client_train(self, client_idx, train_datasets, validation_datasets, **kwargs):
        """client train

        Args:
            client_idx (int): client index
            train_datasets (list): train data for each client
            validation_datasets (list): validation data for each client
        """
        train_info = self.clients[client_idx].train(
            train_datasets[client_idx], validation_datasets, **kwargs
        )
        train_info["client_id"] = client_idx
        agg_client = AggClient()
        agg_client.num_samples = train_info["num_samples"]
        agg_client.weights = self.clients[client_idx].get_weights()
        with self.lock:
            self.aggregate_clients.append(agg_client)
        return train_info

    def llm_client_train(
        self, client_idx, train_datasets, device_id=0, validation_datasets=None, **kwargs
    ):
        """client train for llm mode
        Args:
            client_idx (int): client index
            train_datasets (list): train data for each client
            device_id (int, optional): gpu id.
            validation_datasets (list, optional): validation data for each client.
        """
        train_info = self.clients[client_idx].train(
            train_data=train_datasets[client_idx],
            valid_data=validation_datasets,
            device_id=device_id,
            client_id=client_idx,
            **kwargs,
        )
        train_info["client_id"] = client_idx
        agg_client = AggClient()
        agg_client.num_samples = train_info["num_samples"]
        agg_client.weights = self.clients[client_idx].get_weights()
        with self.lock:
            self.aggregate_clients.append(agg_client)
        return train_info

    def train(self, train_datasets, **kwargs):
        """train——multi-threading to perform client local training

        Args:
            train_datasets (list): train data for each client
        """
        client_threads = []
        LOGGER.info(f"len(self.clients): {len(self.clients)}")
        for idx in range(self.clients_number):
            client_thread = Thread(
                target=self.client_train,
                args=(idx, train_datasets, None),
                kwargs=kwargs,
            )
            client_thread.start()
            client_threads.append(client_thread)
        for thread in client_threads:
            thread.join()
        LOGGER.info("finish training")

    # pylint: disable=unused-argument
    def llm_train(self, train_datasets, **kwargs):
        """train——multi-threading to perform client local training for llm mode
        Args:
            train_datasets (list): train data for each client
        """
        LOGGER.info(
            f"Starting local training for {self.clients_number} clients "
            f"on round {self.rounds + 1} using {self.gpu_num} GPU(s)"
        )
        task_q = Queue()
        for idx in range(self.clients_number):
            task_q.put(idx)

        def _gpu_worker(gpu_idx: int):
            while not task_q.empty():
                try:
                    client_idx = task_q.get_nowait()
                except Empty:
                    break
                LOGGER.info(f"GPU-{gpu_idx} starts training client {client_idx}")
                # —— train ——
                self.llm_client_train(
                    client_idx=client_idx,
                    train_datasets=train_datasets,
                    validation_datasets=None,
                    device_id=gpu_idx,
                )
                LOGGER.info(f"GPU-{gpu_idx} finished client {client_idx}")
                task_q.task_done()

        workers = []
        for gpu_idx in range(self.gpu_num):
            t = Thread(target=_gpu_worker, args=(gpu_idx,))
            t.start()
            workers.append(t)
        for t in workers:
            t.join()
        LOGGER.info("Finished local training for all clients")

    def send_weights_to_clients(self, global_weights):
        """send weights to clients

        Args:
            global_weights (list): aggregated weights
        """
        for client in self.clients:
            client.set_weights(global_weights)
        LOGGER.info("finish send weights to clients")

    def get_global_model(self):
        """get the global model for evaluation
        After final round training, and aggregation
        the global model can be the first client model

        Returns:
            JobBase: sedna_federated_learning.FederatedLearning
        """
        return self.clients[0]

    def predict(self, test_dataset_file):
        """global test to predict the test dataset

        Args:
            test_dataset_file (list): test data

        Returns:
            list: test result
        """
        test_dataset = None
        if self.mode:
            rename_keys_jsonl(test_dataset_file)
        if isinstance(test_dataset_file, str):
            test_dataset = self.dataset.load_data(test_dataset_file, "eval")
        if isinstance(test_dataset_file, list):
            test_dataset = []
            for file in test_dataset_file:
                test_dataset.append(self.dataset.load_data(file, "eval"))
        assert test_dataset is not None, "test_dataset is None"
        LOGGER.info(" begin predict")
        job = self.get_global_model()
        if self.mode:
            data_input = [str(item) for item in test_dataset.x]
        else:
            data_input = test_dataset.x
        test_res = job.inference(data_input)
        LOGGER.info(" finish predict")
        return test_res
