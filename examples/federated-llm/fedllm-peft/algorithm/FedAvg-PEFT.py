<<<<<<< HEAD
# Copyright 2025 The KubeEdge Authors.
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
from collections import OrderedDict

import numpy as np
from sedna.algorithms.aggregation.aggregation import BaseAggregation
from sedna.common.class_factory import ClassType, ClassFactory
import torch

from core.common.log import LOGGER

@ClassFactory.register(ClassType.FL_AGG, "FedAvg-PEFT")
class FedAvg(BaseAggregation, abc.ABC):
    def __init__(self):
        super(FedAvg, self).__init__()

    def aggregate(self, clients):
        """
        Calculate the average weight according to the number of samples

        Parameters
        ----------
        clients: 
            All clients in federated learning job

        Returns
        -------
        update_weights : Array-like
            final weights use to update model layer
        """

        LOGGER.info("begin aggregation....")
        if not len(clients):
            return self.weights
        self.total_size = sum([c.num_samples for c in clients])
        aggregated = OrderedDict()
        for k, v in clients[0].weights.items():
            v0 = torch.as_tensor(v, dtype=v.dtype if isinstance(v, np.ndarray) else v.dtype,
                                    device=v.device if torch.is_tensor(v) else "cpu")
            agg = torch.zeros_like(v0, dtype=torch.float32)
            for c in clients:
                vc = torch.as_tensor(c.weights[k], dtype=agg.dtype, device=agg.device)
                agg += vc * (c.num_samples / self.total_size)
            aggregated[k] = agg
        self.weights = deepcopy(aggregated)
        LOGGER.info("finish aggregation....")
        return aggregated
=======
version https://git-lfs.github.com/spec/v1
oid sha256:0dd50175f114742bbddb86fb68733eb6f0fe0726d022e4a02bf0c87cd58a3bb6
size 2198
>>>>>>> 9676c3e (ya toh aar ya toh par)
