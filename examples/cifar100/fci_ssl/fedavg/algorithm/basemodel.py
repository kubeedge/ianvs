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
import logging
import keras
import numpy as np
import tensorflow as tf
from sedna.common.class_factory import ClassType, ClassFactory
from model import resnet10

__all__ = ["BaseModel"]
os.environ["BACKEND_TYPE"] = "KERAS"
logging.getLogger().setLevel(logging.INFO)


@ClassFactory.register(ClassType.GENERAL, alias="fedavg-client")
class BaseModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        print(f"kwargs: {kwargs}")
        self.batch_size = kwargs.get("batch_size", 1)
        print(f"batch_size: {self.batch_size}")
        self.epochs = kwargs.get("epochs", 1)
        self.learning_rate = kwargs.get("learning_rate", 0.001)
        self.num_classes = 50
        self.task_size = 50
        self.old_task_id = -1
        self.mean = np.array((0.5071, 0.4867, 0.4408), np.float32).reshape(1, 1, -1)
        self.std = np.array((0.2675, 0.2565, 0.2761), np.float32).reshape(1, 1, -1)
        self.fe = resnet10()
        logging.info(type(self.fe))
        self.classifier = None
        self._init_model()

    def _init_model(self):
        self.fe.compile(
            optimizer="sgd",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        self.fe.call(keras.Input(shape=(32, 32, 3)))
        fe_weights = self.fe.get_weights()
        self.fe_weights_length = len(fe_weights)

    def load(self, model_url=None):
        logging.info(f"load model from {model_url}")

    def _initialize(self):
        logging.info(f"initialize finished")

    def get_weights(self):
        weights = []
        fe_weights = self.fe.get_weights()
        self.fe_weights_length = len(fe_weights)
        clf_weights = self.classifier.get_weights()
        weights.extend(fe_weights)
        weights.extend(clf_weights)
        return weights

    def set_weights(self, weights):
        fe_weights = weights[: self.fe_weights_length]
        clf_weights = weights[self.fe_weights_length :]
        self.fe.set_weights(fe_weights)
        self.classifier.set_weights(clf_weights)

    def save(self, model_path=""):
        logging.info("save model")

    def model_info(self, model_path, result, relpath):
        logging.info("model info")
        return {}

    def build_classifier(self):
        if self.classifier != None:
            new_classifier = keras.Sequential(
                [
                    keras.layers.Dense(
                        self.num_classes, kernel_initializer="lecun_normal"
                    )
                ]
            )
            new_classifier.build(
                input_shape=(None, self.fe.layers[-2].output_shape[-1])
            )
            new_weights = new_classifier.get_weights()
            old_weights = self.classifier.get_weights()
            # weight
            new_weights[0][0 : old_weights[0].shape[0], 0 : old_weights[0].shape[1]] = (
                old_weights[0]
            )
            # bias
            new_weights[1][0 : old_weights[1].shape[0]] = old_weights[1]
            new_classifier.set_weights(new_weights)
            self.classifier = new_classifier
        else:
            logging.info(f"input shape is {self.fe.layers[-2].output_shape[-1]}")
            self.classifier = keras.Sequential(
                [
                    keras.layers.Dense(
                        self.num_classes, kernel_initializer="lecun_normal"
                    )
                ]
            )
            self.classifier.build(
                input_shape=(None, self.fe.layers[-2].output_shape[-1])
            )

        logging.info(f"finish ! initialize classifier {self.classifier.summary()}")

    def train(self, train_data, valid_data, **kwargs):
        optimizer = keras.optimizers.SGD(
            learning_rate=self.learning_rate, momentum=0.9, weight_decay=0.0001
        )
        round = kwargs.get("round", -1)
        task_id = kwargs.get("task_id", -1)
        if self.old_task_id != task_id:
            self.old_task_id = task_id
            self.num_classes = self.task_size * (task_id + 1)
            self.build_classifier()
        data = (train_data["label_x"], train_data["label_y"])
        train_db = self.data_process(data)
        logging.info(train_db)
        all_params = []
        all_params.extend(self.fe.trainable_variables)
        all_params.extend(self.classifier.trainable_variables)
        for epoch in range(self.epochs):
            total_loss = 0
            total_num = 0
            logging.info(f"Epoch {epoch + 1} / {self.epochs}")
            logging.info("-" * 50)
            for x, y in train_db:
                with tf.GradientTape() as tape:
                    logits = self.classifier(self.fe(x, training=True), training=True)
                    loss = tf.reduce_mean(
                        keras.losses.sparse_categorical_crossentropy(
                            y, logits, from_logits=True
                        )
                    )
                grads = tape.gradient(loss, all_params)
                optimizer.apply(grads, all_params)
                total_loss += loss
                total_num += 1

                logging.info(
                    f"train round {round}: Epoch {epoch + 1} avg loss: {total_loss / total_num}"
                )
        logging.info(f"finish round {round} train")
        return {"num_samples": data[0].shape[0]}

    def predict(self, data_files, **kwargs):
        result = {}
        for data in data_files:
            x = np.load(data)
            logging.info(f"predicting {x.shape}")
            mean = np.array((0.5071, 0.4867, 0.4408), np.float32).reshape(1, 1, -1)
            std = np.array((0.2675, 0.2565, 0.2761), np.float32).reshape(1, 1, -1)
            x = (tf.cast(x, dtype=tf.float32) / 255.0 - mean) / std
            pred = self.classifier(self.fe(x, training=False))
            prob = tf.nn.softmax(pred, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)
            result[data] = pred.numpy()
        logging.info("finish predict")
        return result

    def eval(self, data, round, **kwargs):
        total_num = 0
        total_correct = 0
        data = self.data_process(data)
        for i, (x, y) in enumerate(data):
            logits = self.model(x, training=False)
            # prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(logits, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)
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
        # data[0]'shape = (50000, 32,32,3) data[1]'shape = (50000,)
        return (
            tf.data.Dataset.from_tensor_slices((data[0], data[1]))
            .shuffle(100000)
            .map(
                lambda x, y: (
                    (tf.cast(x, dtype=tf.float32) / 255.0 - self.mean) / self.std,
                    tf.cast(y, dtype=tf.int32),
                )
            )
            .batch(self.batch_size)
        )
=======
version https://git-lfs.github.com/spec/v1
oid sha256:b442f44a065c8504a4dd4e0a7dee69549047480a76b1a86cbe420436aa258533
size 8005
>>>>>>> 9676c3e (ya toh aar ya toh par)
