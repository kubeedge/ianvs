import abc
from copy import deepcopy
from collections import OrderedDict

import numpy as np
from sedna.algorithms.aggregation.aggregation import BaseAggregation
from sedna.common.class_factory import ClassType, ClassFactory
import torch

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

        print("begin aggregation....")
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
        print("finish aggregation....")
        return aggregated