import abc
from copy import deepcopy
from typing import List

import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from sedna.algorithms.aggregation.aggregation import BaseAggregation
from sedna.common.class_factory import ClassType, ClassFactory


@ClassFactory.register(ClassType.FL_AGG, "FedAvg")
class FedAvg(BaseAggregation, abc.ABC):
    def __init__(self):
        super(FedAvg, self).__init__()
        self.global_model = self.build(num_classes=100)

    """
    Federated averaging algorithm
    """

    @staticmethod
    def build(num_classes: int):
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(3, 3),
                         activation="relu", strides=(2, 2),
                         input_shape=(32, 32, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(0.25))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation="softmax"))

        model.compile(loss="categorical_crossentropy",
                      optimizer="sgd",
                      metrics=["accuracy"])
        return model

    def predict(self, global_model, test_data, round):
        """
        Predict the test data with global model

        Parameters
        ----------
        global_model : Model
            Global model
        test_data : Array-like
            Test data

        Returns
        -------
        predict : Array-like
            Prediction result
        """
        weights = [np.array(layer) for layer in global_model]
        print(test_data)
        self.global_model.set_weights(weights)
        result = {}
        for data in test_data.x:
            x = np.load(data)
            logits = self.global_model(x, training=False)
            pred = tf.cast(tf.argmax(logits, axis=1), tf.int32)
            result[data] = pred.numpy()
        print("finish predict")
        return result

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
