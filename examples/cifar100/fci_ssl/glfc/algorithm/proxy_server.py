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
import copy
import numpy as np
import tensorflow as tf
import logging
from model import resnet10, resnet18, resnet34, lenet5

logging.getLogger().setLevel(logging.INFO)


class ProxyData:
    def __init__(self):
        self.test_data = []
        self.test_label = []


class ProxyServer:
    def __init__(self, learning_rate, num_classes, **kwargs):
        self.learning_rate = learning_rate

        self.encode_model = lenet5(32, 100)

        self.monitor_dataset = ProxyData()
        self.new_set = []
        self.new_set_label = []
        self.num_classes = num_classes
        self.proto_grad = None

        self.best_perf = 0

        self.num_image = 20
        self.Iteration = 250

        self.build_model()
        self.fe_weights_length = len(self.feature_extractor.get_weights())
        self.classifier = None
        self.best_model_1 = None
        self.best_model_2 = None

    def build_model(self):
        self.feature_extractor = resnet10()
        self.feature_extractor.build(input_shape=(None, 32, 32, 3))
        self.feature_extractor.call(keras.Input(shape=(32, 32, 3)))

    def set_weights(self, weights):
        print(f"set weights {self.num_classes}")
        fe_weights = weights[: self.fe_weights_length]
        clf_weights = weights[self.fe_weights_length :]
        self.feature_extractor.set_weights(fe_weights)
        self.classifier.set_weights(clf_weights)

    def increment_class(self, num_classes):
        print(f"increment class {num_classes}")
        self.num_classes = num_classes
        self._initialize_classifier()

    def _initialize_classifier(self):
        if self.classifier != None:
            new_classifier = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(
                        self.num_classes, kernel_initializer="lecun_normal"
                    )
                ]
            )
            new_classifier.build(
                input_shape=(None, self.feature_extractor.layers[-2].output_shape[-1])
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

            self.classifier = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(
                        self.num_classes, kernel_initializer="lecun_normal"
                    )
                ]
            )
            self.classifier.build(
                input_shape=(None, self.feature_extractor.layers[-2].output_shape[-1])
            )
        self.best_model_1 = (self.feature_extractor, self.classifier)
        logging.info(f"finish ! initialize classifier {self.classifier}")

    def model_back(self):
        return [self.best_model_1, self.best_model_2]

    def dataload(self, proto_grad):
        self._initialize_classifier()
        self.proto_grad = proto_grad
        if len(proto_grad) != 0:
            self.reconstruction()
            self.monitor_dataset.test_data = self.new_set
            self.monitor_dataset.test_label = self.new_set_label
            self.last_perf = 0
            self.best_model_1 = self.best_model_2
        cur_perf = self.monitor()
        logging.info(f"in proxy server, current performance is {cur_perf}")
        if cur_perf > self.best_perf:
            self.best_perf = cur_perf
            self.best_model_2 = (self.feature_extractor, self.classifier)

    def monitor(self):
        correct, total = 0, 0
        for x, y in zip(
            self.monitor_dataset.test_data, self.monitor_dataset.test_label
        ):
            y_pred = self.classifier(self.feature_extractor((x)))

            predicts = tf.argmax(y_pred, axis=-1)
            predicts = tf.cast(predicts, tf.int32)
            logging.info(f"y_pred  {predicts} and y {y}")
            correct += tf.reduce_sum(tf.cast(tf.equal(predicts, y), dtype=tf.int32))
            total += x.shape[0]
        acc = 100 * correct / total
        return acc

    def grad2label(self):
        proto_grad_label = []
        for w_single in self.proto_grad:
            pred = tf.argmin(tf.reduce_sum(w_single[-2], axis=-1), axis=-1)
            proto_grad_label.append(pred)
        return proto_grad_label

    def reconstruction(self):
        self.new_set = []
        self.new_set_label = []
        proto_label = self.grad2label()
        proto_label = np.array(proto_label)
        class_ratio = np.zeros((1, 100))

        for i in proto_label:
            class_ratio[0][i] += 1

        for label_i in range(100):
            if class_ratio[0][label_i] > 0:
                agumentation = []

                grad_index = np.where(proto_label == label_i)
                logging.info(f"grad index : {grad_index} and label is {label_i}")
                for j in range(len(grad_index[0])):
                    grad_true_temp = self.proto_grad[grad_index[0][j]]

                    dummy_data = tf.Variable(
                        np.random.rand(1, 32, 32, 3), trainable=True
                    )
                    label_pred = tf.constant([label_i])

                    opt = keras.optimizers.SGD(learning_rate=0.1)
                    cri = keras.losses.SparseCategoricalCrossentropy()

                    recon_model = copy.deepcopy(self.encode_model)

                    for iter in range(self.Iteration):
                        with tf.GradientTape() as tape0:
                            with tf.GradientTape() as tape1:
                                y_pred = recon_model(dummy_data)
                                loss = cri(label_pred, y_pred)
                            dummy_dy_dx = tape1.gradient(
                                loss, recon_model.trainable_variables
                            )

                            grad_diff = 0
                            for gx, gy in zip(dummy_dy_dx, grad_true_temp):
                                gx = tf.cast(gx, tf.double)
                                gy = tf.cast(gy, tf.double)
                                sub_value = tf.subtract(gx, gy)
                                pow_value = tf.pow(sub_value, 2)
                                grad_diff += tf.reduce_sum(pow_value)
                        grad = tape0.gradient(grad_diff, dummy_data)
                        opt.apply_gradients(zip([grad], [dummy_data]))

                        if iter >= self.Iteration - self.num_image:
                            dummy_data_temp = np.asarray(dummy_data)
                            agumentation.append(dummy_data_temp)

                self.new_set.extend(agumentation)
                self.new_set_label.extend([label_i])
