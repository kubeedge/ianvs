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
import numpy as np
from core.common.constant import ParadigmType
from .federated_learning import FederatedLearning
from sedna.algorithms.aggregation import AggClient 
from core.common.log import LOGGER
from threading import Thread, RLock


class FederatedClassIncrementalLearning(FederatedLearning):
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
        super(FederatedClassIncrementalLearning, self).__init__(workspace, **kwargs)
        self.incremental_rounds = kwargs.get("incremental_rounds", 1)
        # self.task_size = kwargs.get("task_size", 10)
        self.system_metric_info = {}
        self.lock = RLock()
        self.aggregate_clients=[]
        self.train_infos=[]
    
    def get_task_size(self, train_datasets):
        return np.unique([train_datasets[i][1] for i in range(len(train_datasets))]).shape[0]
    
    def task_definition(self, dataset_files, task_id):
        """
        Define the task for the class incremental learning paradigm
        """
        # 1. Partition Dataset 
        train_dataset_files, _ = dataset_files[task_id]
        train_datasets = self.train_data_partition(train_dataset_files)
        task_size = self.get_task_size(train_datasets)
        LOGGER.info(f"task_size: {task_size}")
        # 2. According to setting, to split the label and unlabel data for each task
        need_split_label_unlabel_data = 1.0 - self.fl_data_setting.get("label_data_ratio")   > 1e-6
        if need_split_label_unlabel_data:
            train_datasets = self.split_label_unlabel_data(train_datasets)
        # 3. Return the dataset for each task [{label_data, unlabel_data}, ...]
        return train_datasets, task_size
    
    def split_label_unlabel_data(self, train_datasets):
        label_ratio = self.fl.data_setting.get("label_data_ratio")
        new_train_datasets = []
        for i in range(len(train_datasets)):
            train_dataset_dict = {}
            label_data_number = int(label_ratio * len(train_datasets[i][0]))
            # split dataset into label and unlabel data
            train_dataset_dict['label_x'] = train_datasets[i][0][:label_data_number]
            train_dataset_dict['label_y'] = train_datasets[i][1][:label_data_number]
            train_dataset_dict['unlabel_x'] = train_datasets[i][0][label_data_number:]
            train_dataset_dict['unlabel_y'] = train_datasets[i][1][label_data_number:]
            new_train_datasets.append(train_dataset_dict)
        return new_train_datasets

    def init_client(self):
        self.clients = [self.build_paradigm_job(ParadigmType.FEDERATED_CLASS_INCREMENTAL_LEARNING.value)for _ in range(self.clients_number)]

        
    def run(self):
        self.init_client()
        # split_time = self.rounds // self.task_size  # split the dataset into several tasks
        # print(f'split_time: {split_time}') 
        dataset_files = self._split_dataset(self.incremental_rounds)
        for task_id in range(self.incremental_rounds):
            train_datasets, task_size = self.task_definition(dataset_files, task_id)
            for r in range(self.rounds):
                LOGGER.info(f"Round {r} task id: {task_id}")
                self._train(train_datasets, task_id=task_id, round=r, task_size=task_size)
                global_weights = self.aggregator.aggregate(self.aggregate_clients)
                if hasattr(self.aggregator, "helper_function"):
                    self.helper_function(self.train_infos)
                self.send_weights_to_clients(global_weights)
                self.aggregate_clients.clear()
                self.train_infos.clear()
        test_res = self.predict(self.dataset.test_url)
        return test_res, self.system_metric_info
    

    def train_data_partition(self, train_dataset_file):
        return super().train_data_partition(train_dataset_file)
    
    def client_train(self, client_idx, train_datasets, validation_datasets, **kwargs):
        train_info = self.clients[client_idx].train(train_datasets[client_idx], None, **kwargs)
        train_info['client_id'] = client_idx
        aggClient = AggClient()
        aggClient.num_samples = train_info['num_samples']
        aggClient.weights = self.clients[client_idx].get_weights()
        self.lock.acquire()
        self.aggregate_clients.append(aggClient)
        self.train_infos.append(train_info)
        self.lock.release()

    
    def _train(self, train_datasets, **kwargs):
        client_threads = []
        print(f'len(self.clients): {len(self.clients)}')
        for idx in range(len(self.clients)):
            client_thread = Thread(target=self.client_train, args=(idx, train_datasets, None), kwargs=kwargs)
            client_thread.start()
            client_threads.append(client_thread)
        for t in client_threads:
            t.join()
        LOGGER.info('finish training')
        
    
    def send_weights_to_clients(self, global_weights):
        super().send_weights_to_clients(global_weights)
        
    def helper_function(self,train_infos):
        for i in range(len(self.clients)):
            helper_info = self.aggregator.helper_function(train_infos[i])
            self.clients[i].helper_function(helper_info)
        LOGGER.info('finish helper function')
        
  