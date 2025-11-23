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
import tensorflow as tf
import numpy as np
from keras.src.layers import Dense, MaxPooling2D, Conv2D, Flatten, Dropout
from keras.src.models import Sequential

os.environ["BACKEND_TYPE"] = "KERAS"


class Estimator:
    def __init__(self, **kwargs):
        """Model init"""

        self.model = self.build()
        self.has_init = False

    @staticmethod
    def build():
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
        model.add(Dense(1, activation="softmax"))

        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        return model

    def train(
        self,
        train_data,
        valid_data=None,
        epochs=1,
        batch_size=1,
        learning_rate=0.01,
        validation_split=0.2,
    ):
        """Model train"""
        train_loader = (
            tf.data.Dataset.from_tensor_slices(train_data)
            .shuffle(500000)
            .batch(batch_size)
        )
        history = self.model.fit(train_loader, epochs=int(epochs))
        return {k: list(map(np.float, v)) for k, v in history.history.items()}

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def load_weights(self, model):
        if not os.path.isfile(model):
            return
        return self.model.load_weights(model)

    def predict(self, datas):
        return self.model.predict(datas)

    def evaluate(self, test_data, **kwargs):
        pass

    def load(self, model_url):
        print("load model")

    def save(self, model_path=None):
        """
        save model as a single pb file from checkpoint
        """
        print("save model")
=======
version https://git-lfs.github.com/spec/v1
oid sha256:fc86f39d79c414c1cb93004e45ef18fa66e5f03c65d53ad7746fd387fd4f8b21
size 2901
>>>>>>> 9676c3e (ya toh aar ya toh par)
