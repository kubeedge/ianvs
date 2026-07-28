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
import sys

sys.path.append(".")
sys.path.append("..")
import os
import numpy as np
import keras
import tensorflow as tf
from sedna.common.class_factory import ClassType, ClassFactory
from model import resnet10
from FedCiMatch import FedCiMatch
import logging

os.environ["BACKEND_TYPE"] = "KERAS"
__all__ = ["BaseModel"]
logging.getLogger().setLevel(logging.INFO)


@ClassFactory.register(ClassType.GENERAL, alias="FediCarl-Client")
class BaseModel:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.learning_rate = kwargs.get("learning_rate", 0.001)
        self.epochs = kwargs.get("epochs", 1)
        self.batch_size = kwargs.get("batch_size", 32)
        self.task_size = kwargs.get("task_size", 2)
        self.memory_size = kwargs.get("memory_size", 2000)
        self.num_classes = 50  # the number of class for the first task
        self.FedCiMatch = FedCiMatch(
            self.num_classes,
            self.batch_size,
            self.epochs,
            self.learning_rate,
            self.memory_size,
        )
        self.class_learned = 0

    def get_weights(self):
        print("get weights")
        return self.FedCiMatch.get_weights()

    def set_weights(self, weights):
        print("set weights")
        self.FedCiMatch.set_weights(weights)

    def train(self, train_data, val_data, **kwargs):
        task_id = kwargs.get("task_id", 0)
        round = kwargs.get("round", 1)
        task_size = kwargs.get("task_size", self.task_size)
        logging.info(f"in train: {round} task_id:  {task_id}")
        self.class_learned += self.task_size
        self.FedCiMatch.before_train(task_id, round, train_data, task_size)
        self.FedCiMatch.train(round)
        logging.info(f"update example memory")
        self.FedCiMatch.build_exemplar()
        return {"num_samples": self.FedCiMatch.get_data_size(), "task_id": task_id}

    def predict(self, data_files, **kwargs):
        result = {}
        for data in data_files:
            x = np.load(data)
            logging.info(f"predicting {x.shape}")
            res = self.FedCiMatch.predict(x)
            result[data] = res.numpy()
        print("finish predict")
        return result
