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

import keras
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from sedna.common.class_factory import ClassType, ClassFactory

__all__ = ["BaseModel"]
os.environ["BACKEND_TYPE"] = "KEARS"


@ClassFactory.register(ClassType.GENERAL, alias="fedavg")
class BaseModel:
    def __init__(self, **kwargs):
        self.batch_size = kwargs.get("batch_size", 1)
        print(f"batch_size: {self.batch_size}")
        self.epochs = kwargs.get("epochs", 1)
        self.lr = kwargs.get("lr", 0.001)
        self.model = self.build(num_classes=100)
        self.optimizer = tf.keras.optimizers.SGD(
            learning_rate=self.lr, weight_decay=0.0001
        )
        self._init_model()

    @staticmethod
    def build(num_classes: int):
        model = Sequential()
        model.add(
            Conv2D(
                64,
                kernel_size=(3, 3),
                activation="relu",
                strides=(2, 2),
                input_shape=(32, 32, 3),
            )
        )
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(0.25))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation="softmax"))

        model.compile(
            loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"]
        )
        return model

    def _init_model(self):
        self.model.compile(
            optimizer="sgd",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        x = np.random.rand(1, 32, 32, 3)
        y = np.random.randint(0, 100, 1)

        self.model.fit(x, y, epochs=1)

    def load(self, model_url=None):
        print(f"load model from {model_url}")
        extra_model_path = os.path.basename(model_url) + "/model"
        with zipfile.ZipFile(model_url, "r") as zip_ref:
            zip_ref.extractall(extra_model_path)
        self.model = tf.saved_model.load(extra_model_path)

    def _initialize(self):
        print(f"initialize finished")

    def get_weights(self):
        print(f"get_weights")
        weights = [layer.tolist() for layer in self.model.get_weights()]
        print(len(weights))
        return weights

    def set_weights(self, weights):
        weights = [np.array(layer) for layer in weights]
        self.model.set_weights(weights)
        print("----------finish set weights-------------")

    def save(self, model_path=""):
        print("save model")

    def model_info(self, model_path, result, relpath):
        print("model info")
        return {}

    def train(self, train_data, valid_data, **kwargs):
        round = kwargs.get("round", -1)
        print(f"train data: {train_data[0].shape} {train_data[1].shape}")
        train_db = self.data_process(train_data)
        print(train_db)
        for epoch in range(self.epochs):
            total_loss = 0
            total_num = 0
            print(f"Epoch {epoch + 1} / {self.epochs}")
            print("-" * 50)
            for x, y in train_db:
                with tf.GradientTape() as tape:
                    logits = self.model(x, training=True)
                    y_pred = tf.cast(tf.argmax(logits, axis=1), tf.int32)
                    correct = tf.equal(y_pred, y)
                    correct = tf.cast(correct, dtype=tf.int32)
                    correct = tf.reduce_sum(correct)
                    y = tf.one_hot(y, depth=100)
                    # y = tf.squeeze(y, axis=1)
                    loss = tf.reduce_mean(
                        keras.losses.categorical_crossentropy(
                            y, logits, from_logits=True
                        )
                    )
                print(
                    f"loss is {loss}, correct {correct} total is {x.shape[0]} acc : {correct / x.shape[0]}"
                )
                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply(grads, self.model.trainable_variables)
                total_loss += loss
                total_num += 1

            print(
                f"train round {round}: Epoch {epoch + 1} avg loss: {total_loss / total_num}"
            )
        print(f"finish round {round} train")
        return {"num_samples": train_data[0].shape[0]}

    def predict(self, data, **kwargs):
        result = {}
        mean = np.array((0.5071, 0.4867, 0.4408), np.float32).reshape(1, 1, -1)
        std = np.array((0.2675, 0.2565, 0.2761), np.float32).reshape(1, 1, -1)
        for data in data.x:
            x = np.load(data)
            x = (tf.cast(x, dtype=tf.float32) / 255.0 - mean) / std
            logits = self.model(x, training=False)
            pred = tf.cast(tf.argmax(logits, axis=1), tf.int32)
            result[data] = pred.numpy()
        print("finish predict")
        return result

    def eval(self, data, round, **kwargs):
        total_num = 0
        total_correct = 0
        data = self.data_process(data)
        print(f"in evalute data: {data}")
        for i, (x, y) in enumerate(data):
            logits = self.model(x, training=False)
            pred = tf.argmax(logits, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)
            pred = tf.reshape(pred, y.shape)
            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)
            total_num += x.shape[0]
            total_correct += int(correct)
            print(f"total_correct: {total_correct}, total_num: {total_num}")
        acc = total_correct / total_num
        del total_correct
        print(f"finsih round {round}evaluate, acc: {acc}")
        return acc

    def data_process(self, data, **kwargs):
        mean = np.array((0.5071, 0.4867, 0.4408), np.float32).reshape(1, 1, -1)
        std = np.array((0.2675, 0.2565, 0.2761), np.float32).reshape(1, 1, -1)
        assert data is not None, "data is None"
        # data[0]'shape = (50000, 32,32,3) data[1]'shape = (50000,1)
        return (
            tf.data.Dataset.from_tensor_slices((data[0][:5000], data[1][:5000]))
            .shuffle(100000)
            .map(
                lambda x, y: (
                    (tf.cast(x, dtype=tf.float32) / 255.0 - mean) / std,
                    tf.cast(y, dtype=tf.int32),
                )
            )
            .batch(self.batch_size)
        )
