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

import keras
import tensorflow as tf
import numpy as np
from keras.src.layers import Dense
from resnet import resnet10


class NetWork(keras.Model):
    def __init__(self, num_classes, feature_extractor):
        super(NetWork, self).__init__()
        self.num_classes = num_classes
        self.feature = feature_extractor
        self.fc = Dense(num_classes, activation="softmax")

    def call(self, inputs):
        x = self.feature(inputs)
        x = self.fc(x)
        return x

    def feature_extractor(self, inputs):
        return self.feature.predict(inputs)

    def predict(self, fea_input):
        return self.fc(fea_input)

    def get_config(self):
        return {
            "num_classes": self.num_classes,
            "feature_extractor": self.feature,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def incremental_learning(old_model: NetWork, num_class):
    new_model = NetWork(num_class, resnet10(num_class))
    x = np.random.rand(1, 32, 32, 3)
    y = np.random.randint(0, num_class, 1)
    new_model.compile(
        optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    new_model.fit(x, y, epochs=1)
    print(old_model.fc.units, new_model.fc.units)
    for layer in old_model.layers:
        if hasattr(new_model.feature, layer.name):
            new_model.feature.__setattr__(layer.name, layer)
    if num_class > old_model.fc.units:
        original_use_bias = hasattr(old_model.fc, "bias")
        print("original_use_bias", original_use_bias)
        init_class = old_model.fc.units
        k = new_model.fc.kernel
        new_model.fc.kernel.assign(
            tf.pad(old_model.fc.kernel, [[0, 0], [0, num_class - init_class]])
        ) 
        if original_use_bias:
            new_model.fc.bias.assign(
                tf.pad(old_model.fc.bias, [[0, num_class - init_class]])
            )
    new_model.build((None, 32, 32, 3))
    return new_model


def copy_model(model: NetWork):
    cfg = model.get_config()

    copy_model = model.from_config(cfg)
    return copy_model
