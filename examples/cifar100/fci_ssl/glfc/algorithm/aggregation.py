<<<<<<< HEAD
# Copyright 2021 The KubeEdge Authors.
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

import abc
from copy import deepcopy
from typing import List

import numpy as np
from sedna.algorithms.aggregation.aggregation import BaseAggregation
from sedna.common.class_factory import ClassType, ClassFactory
from proxy_server import ProxyServer


@ClassFactory.register(ClassType.FL_AGG, "FedAvg")
class FedAvg(BaseAggregation, abc.ABC):
    def __init__(self):
        super(FedAvg, self).__init__()
        self.proxy_server = ProxyServer(
            learning_rate=0.01, num_classes=10, test_data=None
        )
        self.task_id = -1
        self.num_classes = 10

    def aggregate(self, clients):
        """
        Calculate the average weight according to the number of samples

        Parameters
        ----------
        clients: List
            All clients in federated learning job

        Returns
        -------
        update_weights : Array-like
            final weights use to update model layer
        """

        print("aggregation....")
        if not len(clients):
            return self.weights
        self.total_size = sum([c.num_samples for c in clients])
        old_weight = [np.zeros(np.array(c).shape) for c in next(iter(clients)).weights]
        updates = []
        for inx, row in enumerate(old_weight):
            for c in clients:
                row += np.array(c.weights[inx]) * c.num_samples / self.total_size
            updates.append(row.tolist())

        self.weights = [np.array(layer) for layer in updates]

        print("finish aggregation....")
        return self.weights

    def helper_function(self, train_infos, **kwargs):
        proto_grad = []
        task_id = -1
        for key, value in train_infos.items():
            if "proto_grad" == key and value is not None:
                for grad_i in value:
                    proto_grad.append(grad_i)
            if "task_id" == key:
                task_id = max(value, task_id)
        self.proxy_server.dataload(proto_grad)
        if task_id > self.task_id:
            self.task_id = task_id
            print(f"incremental num classes is {self.num_classes * (task_id + 1)}")
            self.proxy_server.increment_class(self.num_classes * (task_id + 1))
        self.proxy_server.set_weights(self.weights)
        print(f"finish set weight for proxy server")
        return {"best_old_model": self.proxy_server.model_back()}
=======
version https://git-lfs.github.com/spec/v1
oid sha256:7f85d61883f5940edcd494279a7786d06eed8da2ea41f30f0a5ffb82013d5736
size 2944
>>>>>>> 9676c3e (ya toh aar ya toh par)
