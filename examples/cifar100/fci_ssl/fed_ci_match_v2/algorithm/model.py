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

from typing import List
import tensorflow as tf
import numpy as np
import keras
from keras import layers, Sequential


class Conv2D(keras.layers.Layer):
    def __init__(
        self,
        is_combined: bool,
        alpha: float,
        filter_num,
        kernel_size,
        strides=(1, 1),
        padding: str = "valid",
    ):
        super(Conv2D, self).__init__()
        self.is_combined = is_combined
        self.alpha = tf.Variable(alpha)
        self.conv_local = layers.Conv2D(
            filter_num, kernel_size, strides, padding, kernel_initializer="he_normal"
        )
        self.conv_global = layers.Conv2D(
            filter_num, kernel_size, strides, padding, kernel_initializer="he_normal"
        )

    def call(self, inputs):
        return self.alpha * self.conv_global(inputs) + (
            1 - self.alpha
        ) * self.conv_local(inputs)

    def get_alpha(self):
        return self.alpha

    def set_alpha(self, alpha):
        self.alpha.assign(alpha)

    def get_global_weights(self):
        return self.conv_global.get_weights()

    def set_global_weights(self, global_weights):
        self.conv_global.set_weights(global_weights)

    def get_global_variables(self):
        return self.conv_global.trainable_variables

    def merge_to_local(self):
        new_weight = []
        for w_local, w_global in zip(
            self.conv_local.get_weights(), self.conv_global.get_weights()
        ):
            new_weight.append(self.alpha * w_global + (1 - self.alpha) * w_local)
        self.conv_local.set_weights(new_weight)
        self.alpha.assign(0.0)

    def switch_to_global(self):
        self.conv_global.set_weights(self.conv_local.get_weights())


# Input--conv2D--BN--ReLU--conv2D--BN--ReLU--Output
#      \                              /
#       ------------------------------
class BasicBlock(keras.Model):
    def __init__(self, is_combined: bool, filter_num, stride=1):
        super(BasicBlock, self).__init__()

        self.filter_num = filter_num
        self.stride = stride

        self.conv1 = Conv2D(
            is_combined, 0.0, filter_num, (3, 3), strides=stride, padding="same"
        )
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation("relu")

        self.conv2 = Conv2D(
            is_combined, 0.0, filter_num, (3, 3), strides=1, padding="same"
        )
        self.bn2 = layers.BatchNormalization()

        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(
                Conv2D(is_combined, 0.0, filter_num, (1, 1), strides=stride)
            )
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None):
        # [b, h, w, c]
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)

        identity = self.downsample(inputs)

        output = layers.add([out, identity])
        output = tf.nn.relu(output)

        return output


class ResNet(keras.Model):
    def __init__(self, is_combined: bool, layer_dims):  # [2, 2, 2, 2]
        super(ResNet, self).__init__()

        self.is_combined = is_combined
        self.stem = Sequential(
            [
                Conv2D(is_combined, 0.0, 64, (3, 3), strides=(1, 1)),
                layers.BatchNormalization(),
                layers.Activation("relu"),
                layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
            ]
        )

        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)

        # output: [b, 512, h, w],
        self.avgpool = layers.GlobalAveragePooling2D()

    def call(self, inputs, training=None):
        x = self.stem(inputs, training=training)

        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)

        # [b, c]
        x = self.avgpool(x)
        return x

    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = []
        # may down sample
        res_blocks.append(BasicBlock(self.is_combined, filter_num, stride))
        for _ in range(1, blocks):
            res_blocks.append(BasicBlock(self.is_combined, filter_num, stride=1))
        return Sequential(res_blocks)

    def get_alpha(self):
        convs = self._get_all_conv_layers()
        ret = []
        for conv in convs:
            ret.append(conv.get_alpha())
        return ret

    def set_alpha(self, alpha=0.0):
        convs = self._get_all_conv_layers()
        for conv in convs:
            conv.set_alpha(alpha)

    def merge_to_local_model(self):
        convs = self._get_all_conv_layers()
        for conv in convs:
            conv.merge_to_local()

    def switch_to_global(self):
        convs = self._get_all_conv_layers()
        for conv in convs:
            conv.switch_to_global()

    def initialize_alpha(self):
        convs = self._get_all_conv_layers()
        for conv in convs:
            conv.set_alpha(np.random.random())

    def set_global_model(self, global_model):
        local_convs = self._get_all_conv_layers()
        global_convs = global_model._get_all_conv_layers()
        for local_conv, global_conv in zip(local_convs, global_convs):
            local_conv.set_global_weights(global_conv.get_global_weights())

    def get_global_variables(self):
        convs = self._get_all_conv_layers()
        ret = []
        for conv in convs:
            ret.extend(conv.get_global_variables())
        return ret

    def _get_all_conv_layers(self) -> List[Conv2D]:
        def get_all_conv_layers_(model):
            convs = []
            for i in model.layers:
                if isinstance(i, Conv2D):
                    convs.append(i)
                elif isinstance(i, keras.Model):
                    convs.extend(get_all_conv_layers_(i))
            return convs

        return get_all_conv_layers_(self)


def resnet10(is_combined=False) -> ResNet:
    return ResNet(is_combined, [1, 1, 1, 1])


def resnet18(is_combined=False) -> ResNet:
    return ResNet(is_combined, [2, 2, 2, 2])


def resnet34(is_combined=False) -> ResNet:
    return ResNet(is_combined, [3, 4, 6, 3])


class LeNet5(keras.Model):
    def __init__(self):  # [2, 2, 2, 2]
        super(LeNet5, self).__init__()
        self.cnn_layers = keras.Sequential(
            [
                Conv2D(True, 0.0, 6, kernel_size=(5, 5), padding="valid"),
                layers.ReLU(),
                layers.MaxPool2D(pool_size=(2, 2)),
                Conv2D(True, 0.0, 16, kernel_size=(5, 5), padding="valid"),
                layers.ReLU(),
                layers.MaxPool2D(pool_size=(2, 2)),
            ]
        )

        self.flatten = layers.Flatten()

        self.fc_layers = keras.Sequential(
            [
                layers.Dense(120),
                layers.ReLU(),
                layers.Dense(84),
                layers.ReLU(),
            ]
        )

    def call(self, inputs, training=None):
        x = self.cnn_layers(inputs, training=training)

        x = self.flatten(x, training=training)
        x = self.fc_layers(x, training=training)

    def get_alpha(self):
        convs = self._get_all_conv_layers()
        ret = []
        for conv in convs:
            ret.append(conv.get_alpha())
        return ret

    def set_alpha(self, alpha=0.0):
        convs = self._get_all_conv_layers()
        for conv in convs:
            conv.set_alpha(alpha)

    def merge_to_local_model(self):
        convs = self._get_all_conv_layers()
        for conv in convs:
            conv.merge_to_local()

    def switch_to_global(self):
        convs = self._get_all_conv_layers()
        for conv in convs:
            conv.switch_to_global()

    def initialize_alpha(self):
        convs = self._get_all_conv_layers()
        for conv in convs:
            conv.set_alpha(np.random.random())

    def set_global_model(self, global_model):
        local_convs = self._get_all_conv_layers()
        global_convs = global_model._get_all_conv_layers()
        for local_conv, global_conv in zip(local_convs, global_convs):
            local_conv.set_global_weights(global_conv.get_global_weights())

    def get_global_variables(self):
        convs = self._get_all_conv_layers()
        ret = []
        for conv in convs:
            ret.extend(conv.get_global_variables())
        return ret

    def _get_all_conv_layers(self) -> List[Conv2D]:
        def get_all_conv_layers_(model):
            convs = []
            for i in model.layers:
                if isinstance(i, Conv2D):
                    convs.append(i)
                elif isinstance(i, keras.Model):
                    convs.extend(get_all_conv_layers_(i))
            return convs

        return get_all_conv_layers_(self)
=======
version https://git-lfs.github.com/spec/v1
oid sha256:67200e5eef3d82281287ca2da85c20576527d6f2039023e166f0cb687bf5af71
size 9740
>>>>>>> 9676c3e (ya toh aar ya toh par)
