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

import os
import numpy as np
import keras
import tensorflow as tf
from sedna.common.class_factory import ClassType, ClassFactory
from model import resnet10, lenet5
from GLFC import GLFC_Client
import logging

os.environ["BACKEND_TYPE"] = "KERAS"
__all__ = ["BaseModel"]
logging.getLogger().setLevel(logging.INFO)


@ClassFactory.register(ClassType.GENERAL, alias="GLFCMatch-Client")
class BaseModel:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.learning_rate = kwargs.get("learning_rate", 0.001)
        self.epochs = kwargs.get("epochs", 1)
        self.batch_size = kwargs.get("batch_size", 32)
        self.task_size = kwargs.get("task_size", 10)
        self.memory_size = kwargs.get("memory_size", 2000)
        self.encode_model = lenet5(32, 100)
        self.encode_model.call(keras.Input(shape=(32, 32, 3)))
        self.num_classes = 10  # the number of class for the first task
        self.GLFC_Client = GLFC_Client(
            self.num_classes,
            self.batch_size,
            self.task_size,
            self.memory_size,
            self.epochs,
            self.learning_rate,
            self.encode_model,
        )
        self.best_old_model = []
        self.class_learned = 0
        self.fe_weights_length = len(self.GLFC_Client.feature_extractor.get_weights())

    def get_weights(self):
        print("get weights")
        weights = []
        fe_weights = self.GLFC_Client.feature_extractor.get_weights()
        clf_weights = self.GLFC_Client.classifier.get_weights()
        weights.extend(fe_weights)
        weights.extend(clf_weights)
        return weights

    def set_weights(self, weights):
        print("set weights")
        fe_weights = weights[: self.fe_weights_length]

        clf_weights = weights[self.fe_weights_length :]
        self.GLFC_Client.feature_extractor.set_weights(fe_weights)
        self.GLFC_Client.classifier.set_weights(clf_weights)

    def train(self, train_data, val_data, **kwargs):
        task_id = kwargs.get("task_id", 0)
        round = kwargs.get("round", 1)
        logging.info(f"in train: {round} task_id:  {task_id}")
        self.class_learned += self.task_size
        self.GLFC_Client.before_train(
            task_id, train_data, self.class_learned, old_model=self.best_old_model
        )

        self.GLFC_Client.train(round)
        proto_grad = self.GLFC_Client.proto_grad()
        return {
            "num_samples": self.GLFC_Client.get_data_size(),
            "proto_grad": proto_grad,
            "task_id": task_id,
        }

    def helper_function(self, helper_info, **kwargs):
        self.best_old_model = helper_info["best_old_model"]
        if self.best_old_model[1] != None:
            self.GLFC_Client.old_model = self.best_old_model[1]
        else:
            self.GLFC_Client.old_model = self.best_old_model[0]

    def predict(self, datas, **kwargs):
        result = {}
        mean = np.array((0.5071, 0.4867, 0.4408), np.float32).reshape(1, 1, -1)
        std = np.array((0.2675, 0.2565, 0.2761), np.float32).reshape(1, 1, -1)
        for data in datas:
            x = np.load(data)
            x = (tf.cast(x, dtype=tf.float32) / 255.0 - mean) / std
            logits = self.GLFC_Client.model_call(x, training=False)
            pred = tf.cast(tf.argmax(logits, axis=1), tf.int32)
            result[data] = pred.numpy()
        print("finish predict")
        return result
=======
version https://git-lfs.github.com/spec/v1
oid sha256:df57129fd53e4a096c85a48004ebbc1fd5864ba0240d7e0cb382c7a81c0501b1
size 4026
>>>>>>> 9676c3e (ya toh aar ya toh par)
