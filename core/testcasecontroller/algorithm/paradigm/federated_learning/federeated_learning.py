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

"""Federated Class-Incremental Semi-Supervised  Learning Paradigm"""
import threading
import multiprocessing as mp
import asyncio 
from multiprocessing import Process
from sedna.service.server import AggregationServer

from core.common.constant import ParadigmType, ModuleType
from core.common.utils import get_file_format
from core.testcasecontroller.algorithm.paradigm.base import ParadigmBase
from core.testenvmanager.dataset.utils import read_data_from_file_to_npy, partition_data

class FederatedLearning(ParadigmBase):
    # pylint: disable=too-many-locals
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

    LOCAL_HOST = '127.0.0.1'

    def __init__(self, workspace, **kwargs):
        ParadigmBase.__init__(self, workspace, **kwargs)

        super(FederatedLearning, self).__init__(workspace, **kwargs)
        self.workspace = workspace
        self.kwargs = kwargs

        self.fl_data_setting = kwargs.get("fl_data_setting")
        self.backend = kwargs.get("backend")
        self.global_model = None  # global model to perform global evaluation
        self.rounds = kwargs.get("round", 1)
        print(self.rounds)
        self.clients = []
        self.clients_number = kwargs.get("client_number", 1)
        self.aggregation, self.aggregator = self.module_instances.get(ModuleType.AGGREGATION.value)

    def run_server(self):
        aggregation_algorithm = self.aggregation
        exit_round = self.rounds
        participants_count = self.clients_number
        print("running server!!!!")
        server = AggregationServer(
            aggregation=aggregation_algorithm,
            exit_round=exit_round,
            ws_size=1000 * 1024 * 1024,
            participants_count=participants_count,
            host=self.LOCAL_HOST

        )
        server.start()

    def init_client(self):
        self.clients = [self.build_paradigm_job(ParadigmType.FEDERATED_LEARNING.value) for i in
                        range(self.clients_number)]

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

        server_thead = threading.Thread(target=self.run_server)
        server_thead.start()
        print(f"server is start and server is alive:  {server_thead.is_alive()}")
        # self.init_client()
        rounds = self.rounds
        dataset_files = self._split_dataset(1) # only one split ——all the data
        train_dataset_file, eval_dataset_file = dataset_files[0]
        train_datasets = self.train_data_partition(train_dataset_file)
        # for r in range(rounds):
        #     print(f"Round {r} train dataset: {train_dataset_file}")
        self._train(train_datasets, rounds=rounds)
        print(f'finish trianing for fedavg')
        server_thead.join()
        test_res = self.predict(self.dataset.test_url, "test")
        return test_res, self.system_metric_info

    def _split_dataset(self, splitting_dataset_times=1):
        train_dataset_ratio = self.fl_data_setting.get("train_ratio")
        splitting_dataset_method = self.fl_data_setting.get("splitting_method")
        return self.dataset.split_dataset(self.dataset.train_url,
                                          get_file_format(self.dataset.train_url),
                                          train_dataset_ratio,
                                          method=splitting_dataset_method,
                                          dataset_types=("model_train", "model_eval"),
                                          output_dir=self.dataset_output_dir(),
                                          times=splitting_dataset_times)

    def train_data_partition(self, train_dataset_file):
        """
         Partition the dataset for the class incremental learning paradigm
         - i.i.d
         - non-i.i.d
         """
        print(train_dataset_file)
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
        # TODO Partition data to iid or non-iid
        train_datasets = partition_data(train_datasets, self.clients_number,
                                                     self.fl_data_setting.get("data_partition"))
        return train_datasets

    def client_train(self, train_datasets, validation_datasets, post_process, **kwargs):
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        client = self.build_paradigm_job(ParadigmType.FEDERATED_LEARNING.value)
        
        client.train(train_datasets, validation_datasets, post_process, **kwargs)
        loop.close()
        self.clients.append(client)
        
    def _train(self, train_datasets, **kwargs):
        
        mp.set_start_method('spawn')
      
        # config = {"round": round}
        clients_threads = []
        for i in range(self.clients_number):
            print(i , self.clients_number)
            # config = {
            #     "round": round,
            # }
            # self.clients[i].train(train_datasets[i], None, None, **kwargs)
            t = threading.Thread(target=self.client_train, args=(train_datasets[i], None, None), kwargs=kwargs)
            # t = Process(target=self.clients[i].train, args=(train_datasets[i], None, None), kwargs=kwargs)

            clients_threads.append(t)
            t.start()
        for t in clients_threads:
            print(f"client process is alive: {t.is_alive()}")
            t.join()
            print(f"finish training {t}")
        return

    def local_eval(self, train_dataset_file, round):
        """
        Evaluate the model on the local dataset
        """
        train_dataset = None
        if isinstance(train_dataset_file, str):
            train_dataset = self.dataset.load_data(train_dataset_file, "train")
        if isinstance(train_dataset_file, list):
            train_dataset = []
            for file in train_dataset_file:
                train_dataset.append(self.dataset.load_data(file, "train"))
        assert train_dataset is not None, "train_dataset is None"
        train_dataset = read_data_from_file_to_npy(train_dataset)
        for client in self.clients:
            client.evaluate(train_dataset, round=round)
        print('finish local eval')

    def get_global_model(self):
        self.global_model = self.clients[0]
        return self.global_model

    def predict(self, test_dataset_file, rounds):
        # global test
        test_dataset = None
        if isinstance(test_dataset_file, str):
            test_dataset = self.dataset.load_data(test_dataset_file, "eval")
        if isinstance(test_dataset_file, list):
            test_dataset = []
            for file in test_dataset_file:
                test_dataset.append(self.dataset.load_data(file, "eval"))
        assert test_dataset is not None, "test_dataset is None"
        job = self.get_global_model()
        test_res = job.inference(test_dataset, rounds=rounds)
        print(f" after predict {len(test_res)}")
        return test_res


class FederatedClassIncrementalLearning(FederatedLearning):
    EXECUTION_ROUND = 1
    def __init__(self, workspace, **kwargs):
        super(FederatedClassIncrementalLearning, self).__init__(workspace, **kwargs)
        self.rounds = kwargs.get("incremental_rounds", 1)
        self.task_size = kwargs.get("task_size", 10)

    def task_definition(self):
        """
        Define the task for the class incremental learning paradigm
        """
        pass
    pass