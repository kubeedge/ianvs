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
import zipfile
import logging
import keras
import numpy as np
import tensorflow as tf
from sedna.common.class_factory import ClassType, ClassFactory
from resnet import resnet10
from network import NetWork, incremental_learning

__all__ = ["BaseModel"]
os.environ["BACKEND_TYPE"] = "KERAS"
logging.getLogger().setLevel(logging.INFO)


@ClassFactory.register(ClassType.GENERAL, alias="fcil")
class BaseModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        print(f"kwargs: {kwargs}")
        self.batch_size = kwargs.get("batch_size", 1)
        print(f"batch_size: {self.batch_size}")
        self.epochs = kwargs.get("epochs", 1)
        self.lr = kwargs.get("lr", 0.001)
        self.optimizer = keras.optimizers.SGD(learning_rate=self.lr)
        self.old_task_id = -1
        self.fe = resnet10(10)
        logging.info(type(self.fe))
        self.model = NetWork(100, self.fe)
        self._init_model()

    def _init_model(self):
        self.model.compile(
            optimizer="sgd",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        x = np.random.rand(1, 32, 32, 3)
        y = np.random.randint(0, 10, 1)

        self.model.fit(x, y, epochs=1)

    def load(self, model_url=None):
        logging.info(f"load model from {model_url}")
        extra_model_path = os.path.basename(model_url) + "/model"
        with zipfile.ZipFile(model_url, "r") as zip_ref:
            zip_ref.extractall(extra_model_path)
        self.model = tf.saved_model.load(extra_model_path)

    def _initialize(self):
        logging.info(f"initialize finished")

    def get_weights(self):
        logging.info(f"get_weights")
        weights = [layer.tolist() for layer in self.model.get_weights()]
        logging.info(len(weights))
        return weights

    def set_weights(self, weights):
        weights = [np.array(layer) for layer in weights]
        self.model.set_weights(weights)
        logging.info("----------finish set weights-------------")

    def save(self, model_path=""):
        logging.info("save model")

    def model_info(self, model_path, result, relpath):
        logging.info("model info")
        return {}

    def train(self, train_data, valid_data, **kwargs):
        round = kwargs.get("round", -1)
        self.model.compile(
            optimizer=self.optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        logging.info(
            f"train data: {train_data['label_x'].shape} {train_data['label_y'].shape}"
        )
        train_db = self.data_process(train_data)
        logging.info(train_db)
        for epoch in range(self.epochs):
            total_loss = 0
            total_num = 0
            logging.info(f"Epoch {epoch + 1} / {self.epochs}")
            logging.info("-" * 50)
            for x, y in train_db:
                with tf.GradientTape() as tape:
                    logits = self.model(x, training=True)
                    loss = tf.reduce_mean(
                        keras.losses.sparse_categorical_crossentropy(
                            y, logits, from_logits=True
                        )
                    )
                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply(grads, self.model.trainable_variables)
                # self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                total_loss += loss
                total_num += 1

                logging.info(
                    f"train round {round}: Epoch {epoch + 1} avg loss: {total_loss / total_num}"
                )
        logging.info(f"finish round {round} train")
        self.eval(train_data, round)
        return {"num_samples": train_data["label_x"].shape[0]}

    def predict(self, data, **kwargs):
        result = {}
        for data in data.x:
            x = np.load(data)
            logits = self.model(x, training=False)
            pred = tf.cast(tf.argmax(logits, axis=1), tf.int32)
            result[data] = pred.numpy()
        logging.info("finish predict")
        return result

    def eval(self, data, round, **kwargs):
        total_num = 0
        total_correct = 0
        data = self.data_process(data)
        for i, (x, y) in enumerate(data):
            logits = self.model(x, training=False)
            pred = tf.argmax(logits, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)
            pred = tf.reshape(pred, y.shape)
            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)
            total_num += x.shape[0]
            total_correct += int(correct)
        logging.info(f"total_correct: {total_correct}, total_num: {total_num}")
        acc = total_correct / total_num
        del total_correct
        logging.info(f"finsih round {round}evaluate, acc: {acc}")
        return acc

    def data_process(self, data, **kwargs):

        assert data is not None, "data is None"
        x_trian = data["label_x"]
        y_train = data["label_y"]
        # data[0]'shape = (50000, 32,32,3) data[1]'shape = (50000,1)
        return (
            tf.data.Dataset.from_tensor_slices((x_trian, y_train))
            .shuffle(100000)
            .batch(self.batch_size)
        )
=======
version https://git-lfs.github.com/spec/v1
oid sha256:c9d14a7d10c3ff029e324201920cdbc01d8e28e1962704a08b3672886d7c4c65
size 5920
>>>>>>> 9676c3e (ya toh aar ya toh par)
