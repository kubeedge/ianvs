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
import torch
from sedna.algorithms.aggregation.aggregation import BaseAggregation
from sedna.common.class_factory import ClassType, ClassFactory


@ClassFactory.register(ClassType.FL_AGG, "FedAvgM-PEFT")
class FedAvgM(BaseAggregation, abc.ABC):
    """
    FedAvg with server momentum (Hsu et al., 2019, arXiv:1909.06335)
    """
    def __init__(self, beta: float = 0.7, server_lr: float = 1.0):
        super(FedAvgM, self).__init__()
        self.beta = beta
        self.server_lr = server_lr
        self.moments = None

    def aggregate(self, clients):
        if not clients:
            return self.weights

        if self.weights is None:
            self.weights = deepcopy(clients[0].weights)

        total_size = sum(c.num_samples for c in clients)

        if self.moments is None:
            self.moments = {
                k: torch.zeros_like(
                        torch.as_tensor(v, dtype=torch.float32, device="cpu"))
                for k, v in self.weights.items()
            }

        aggregated = OrderedDict()
        for k, w_t_raw in self.weights.items():
            w_t = torch.as_tensor(w_t_raw, dtype=torch.float32, device="cpu")

            g_t = torch.zeros_like(w_t)
            for c in clients:
                w_i = torch.as_tensor(c.weights[k], dtype=torch.float32, device="cpu")
                g_t += (c.num_samples / total_size) * (w_t - w_i)

            self.moments[k] = self.beta * self.moments[k] + g_t

            w_new = w_t - self.server_lr * self.moments[k]
            aggregated[k] = w_new

        self.weights = deepcopy(aggregated)
        return aggregated
