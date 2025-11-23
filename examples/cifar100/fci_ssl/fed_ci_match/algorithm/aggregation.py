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

import numpy as np
from sedna.algorithms.aggregation.aggregation import BaseAggregation
from sedna.common.class_factory import ClassType, ClassFactory

@ClassFactory.register(ClassType.FL_AGG, "FedAvg")
class FedAvg(BaseAggregation, abc.ABC):
    def __init__(self):
        super(FedAvg, self).__init__()

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
        old_weight = [np.zeros(np.array(c).shape) for c in
                      next(iter(clients)).weights]
        updates = []
        for inx, row in enumerate(old_weight):
            for c in clients:
                row += (np.array(c.weights[inx]) * c.num_samples
                        / self.total_size)
            updates.append(row.tolist())
        
        print("finish aggregation....")
        return [np.array(layer) for layer in updates]
=======
version https://git-lfs.github.com/spec/v1
oid sha256:4c4f3d39835fe063709bccf26ccc1ac0142c92d6b4a55b3166a859a35b852837
size 1887
>>>>>>> 9676c3e (ya toh aar ya toh par)
