import abc
from copy import deepcopy
from typing import List

import numpy as np
import tensorflow as tf
from sedna.algorithms.aggregation.aggregation import BaseAggregation
from sedna.common.class_factory import ClassType, ClassFactory

@ClassFactory.register(ClassType.FL_AGG, "FedAvg")
class FedAvg(BaseAggregation, abc.ABC):
    def __init__(self):
        super(FedAvg, self).__init__()
    """
    Federated averaging algorithm
    """

    def aggregate(self, clients:List):
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
        # print(next(iter(clients)).weights)
        old_weight = [np.zeros(np.array(c).shape) for c in
                      next(iter(clients)).weights]
        updates = []
        for inx, row in enumerate(old_weight):
            for c in clients:
                row += (np.array(c.weights[inx]) * c.num_samples
                        / self.total_size)
            updates.append(row.tolist())
        self.weights = deepcopy(updates)
        print("finish aggregation....")
        return updates
