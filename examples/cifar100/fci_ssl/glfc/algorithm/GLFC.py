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

import copy
import numpy as np
import tensorflow as tf
import keras
import logging
from agumentation import *
from data_prepocessor import *
from model import resnet10


def get_one_hot(target, num_classes):
    y = tf.one_hot(target, depth=num_classes)
    if len(y.shape) == 3:
        y = tf.squeeze(y, axis=1)
    return y


class GLFC_Client:
    def __init__(
        self,
        num_classes,
        batch_size,
        task_size,
        memory_size,
        epochs,
        learning_rate,
        encode_model,
    ):
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.encode_model = encode_model

        self.num_classes = num_classes
        logging.info(f"num_classes is {num_classes}")
        self.batch_size = batch_size
        self.task_size = task_size

        self.old_model = None
        self.train_set = None

        self.exemplar_set = []  #
        self.class_mean_set = []
        self.learned_classes = []
        self.learned_classes_numebr = 0
        self.memory_size = memory_size

        self.old_task_id = -1
        self.current_classes = None
        self.last_class = None
        self.train_loader = None
        self.build_feature_extractor()
        self.classifier = None
        self.labeled_train_set = None
        self.unlabeled_train_set = None
        self.data_preprocessor = Dataset_Preprocessor(
            "cifar100", Weak_Augment("cifar100"), RandAugment("cifar100")
        )
        self.warm_up_epochs = 10

    def build_feature_extractor(self):
        self.feature_extractor = resnet10()
        self.feature_extractor.build(input_shape=(None, 32, 32, 3))
        self.feature_extractor.call(keras.Input(shape=(32, 32, 3)))
        self.feature_extractor.load_weights(
            "examples/cifar100/fci_ssl/glfc/algorithm/feature_extractor.weights.h5"
        )

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
            logging.info(
                f"input shape is {self.feature_extractor.layers[-2].output_shape[-1]}"
            )
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

        logging.info(f"finish ! initialize classifier {self.classifier.summary()}")

    def before_train(self, task_id, train_data, class_learned, old_model):
        logging.info(f"------before train task_id: {task_id}------")
        self.need_update = task_id != self.old_task_id
        if self.need_update:
            self.old_task_id = task_id
            self.num_classes = self.task_size * (task_id + 1)
            if self.current_classes is not None:
                self.last_class = self.current_classes
            logging.info(
                f"self.last_class is , {self.last_class}, {self.num_classes} tasksize is {self.task_size}, task_id is {task_id}"
            )
            self._initialize_classifier()
            self.current_classes = np.unique(train_data["label_y"]).tolist()
            self.update_new_set(self.need_update)
            self.labeled_train_set = (train_data["label_x"], train_data["label_y"])
            self.unlabeled_train_set = (
                train_data["unlabel_x"],
                train_data["unlabel_y"],
            )
        if len(old_model) != 0:
            self.old_model = old_model[0]
        self.labeled_train_set = (train_data["label_x"], train_data["label_y"])
        self.unlabeled_train_set = (train_data["unlabel_x"], train_data["unlabel_y"])
        self.labeled_train_loader, self.unlabeled_train_loader = self._get_train_loader(
            True
        )
        logging.info(
            f"------finish before train task_id: {task_id} {self.current_classes}------"
        )

    def update_new_set(self, need_update):
        if need_update and self.last_class is not None:
            # update exemplar
            self.learned_classes += self.last_class
            self.learned_classes_numebr += len(self.last_class)
            m = int(self.memory_size / self.learned_classes_numebr)
            self._reduce_exemplar_set(m)
            for i in self.last_class:
                images = self.get_train_set_data(i)
                self._construct_exemplar_set(images, i, m)

    def _get_train_loader(self, mix):
        self.mean = np.array((0.5071, 0.4867, 0.4408), np.float32).reshape(1, 1, -1)
        self.std = np.array((0.2675, 0.2565, 0.2761), np.float32).reshape(1, 1, -1)
        train_x = self.labeled_train_set[0]
        train_y = self.labeled_train_set[1]
        if mix:
            for exm_set in self.exemplar_set:
                logging.info(f"mix the exemplar{len(exm_set[0])}, {len(exm_set[1])}")
                label = np.array(exm_set[1])
                train_x = np.concatenate((train_x, exm_set[0]), axis=0)
                train_y = np.concatenate((train_y, label), axis=0)
        label_data_loader = self.data_preprocessor.preprocess_labeled_dataset(
            train_x, train_y, self.batch_size
        )
        unlabel_data_loader = None
        if len(self.unlabeled_train_set[0]) > 0:
            unlabel_data_loader = self.data_preprocessor.preprocess_unlabeled_dataset(
                self.unlabeled_train_set[0],
                self.unlabeled_train_set[1],
                self.batch_size,
            )
            logging.info(
                f"unlabel_x shape: {self.unlabeled_train_set[0].shape} and unlabel_y shape: {self.unlabeled_train_set[1].shape}"
            )
        return (label_data_loader, unlabel_data_loader)

    def train(self, round):
        opt = keras.optimizers.Adam(
            learning_rate=self.learning_rate, weight_decay=0.00001
        )
        feature_extractor_params = self.feature_extractor.trainable_variables
        classifier_params = self.classifier.trainable_variables
        all_params = []
        all_params.extend(feature_extractor_params)
        all_params.extend(classifier_params)

        for epoch in range(self.epochs):
            # following code is for semi-supervised learning
            # for labeled_data, unlabeled_data in zip(
            #     self.labeled_train_loader, self.unlabeled_train_loader
            # ):
            #     labeled_x, labeled_y = labeled_data
            #     unlabeled_x, weak_unlabeled_x, strong_unlabeled_x, unlabeled_y = (
            #         unlabeled_data
            #     )

            # following code is for supervised learning
            for step, (x, y) in enumerate(self.labeled_train_loader):
                # opt = keras.optimizers.SGD(learning_rate=self.learning_rate, weight_decay=0.00001)
                with tf.GradientTape() as tape:
                    supervised_loss = self._compute_loss(x, y)
                    loss = supervised_loss

                    # following code is for semi-supervised learning
                    # if epoch > self.warm_up_epochs:
                    #     unsupervised_loss = self.unsupervised_loss(
                    #         weak_unlabeled_x, strong_unlabeled_x
                    #     )
                    #     loss = loss + 0.5 * unsupervised_loss
                    #     logging.info(
                    #         f"supervised loss is {supervised_loss} unsupervised loss is {unsupervised_loss}"
                    #     )
                logging.info(
                    f"------round{round} epoch{epoch}  loss: {loss} and loss dim is {loss.shape}------"
                )
                grads = tape.gradient(loss, all_params)
                opt.apply_gradients(zip(grads, all_params))

        logging.info(f"------finish round{round} traning------")

    def model_call(self, x, training=False):
        input = self.feature_extractor(inputs=x, training=training)
        return self.classifier(inputs=input, training=training)

    def _compute_loss(self, imgs, labels):
        logging.info(f"self.old_model is available: {self.old_model is not None}")
        y_pred = self.model_call(imgs, training=True)
        target = get_one_hot(labels, self.num_classes)
        logits = y_pred
        pred = tf.argmax(logits, axis=1)
        pred = tf.cast(pred, dtype=tf.int32)
        pred = tf.reshape(pred, labels.shape)

        y = tf.cast(labels, dtype=tf.int32)
        correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
        correct = tf.reduce_sum(correct)
        logging.info(
            f"current class numbers is {self.num_classes} correct is {correct} and acc is {correct/imgs.shape[0]} tasksize is {self.task_size} self.old_task_id {self.old_task_id}"
        )
        if self.old_model == None:
            w = self.efficient_old_class_weight(target, labels)
            loss = tf.reduce_mean(
                keras.losses.categorical_crossentropy(target, y_pred, from_logits=True)
                * w
            )
            logging.info(
                f"in _compute_loss, without old model loss is {loss} and shape is {loss.shape}"
            )
            return loss
        else:
            w = self.efficient_old_class_weight(target, labels)
            loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(target, y_pred, from_logits=True) * w
            )
            distill_target = tf.Variable(get_one_hot(labels, self.num_classes))
            old_target = tf.sigmoid(self.old_model[1](self.old_model[0]((imgs))))
            old_task_size = old_target.shape[1]
            distill_target[:, :old_task_size].assign(old_target)
            loss_old = tf.reduce_mean(
                keras.losses.binary_crossentropy(
                    distill_target, y_pred, from_logits=True
                )
            )
            logging.info(f"loss old  is {loss_old}")
            return loss + loss_old

    def unsupervised_loss(self, weak_x, strong_x):
        self.accept_threshold = 0.95
        prob_on_wux = tf.nn.softmax(
            self.classifier(
                self.feature_extractor(weak_x, training=True), training=True
            )
        )
        pseudo_mask = tf.cast(
            (tf.reduce_max(prob_on_wux, axis=1) >= self.accept_threshold), tf.float32
        )
        pse_uy = tf.one_hot(
            tf.argmax(prob_on_wux, axis=1), depth=self.num_classes
        ).numpy()
        prob_on_sux = tf.nn.softmax(
            self.classifier(
                self.feature_extractor(strong_x, training=True), training=True
            )
        )
        loss = keras.losses.categorical_crossentropy(pse_uy, prob_on_sux)
        loss = tf.reduce_mean(loss * pseudo_mask)
        return loss

    def efficient_old_class_weight(self, output, labels):
        pred = tf.sigmoid(output)
        N, C = pred.shape
        class_mask = tf.zeros([N, C], dtype=tf.float32)
        class_mask = tf.Variable(class_mask)
        ids = np.zeros([N, 2], dtype=np.int32)
        for i in range(N):
            ids[i][0] = i
            ids[i][1] = labels[i]
        updates = tf.ones([N], dtype=tf.float32)
        class_mask = tf.tensor_scatter_nd_update(class_mask, ids, updates)
        target = get_one_hot(labels, self.num_classes)
        g = tf.abs(target - pred)
        g = tf.reduce_sum(g * class_mask, axis=1)
        idx = tf.cast(tf.reshape(labels, (-1, 1)), tf.int32)
        if len(self.learned_classes) != 0:
            for i in self.learned_classes:
                mask = tf.math.not_equal(idx, i)
                negative_value = tf.cast(tf.fill(tf.shape(idx), -1), tf.int32)
                idx = tf.where(mask, idx, negative_value)
            index1 = tf.cast(tf.equal(idx, -1), tf.float32)
            index2 = tf.cast(tf.not_equal(idx, -1), tf.float32)
            w1 = tf.where(
                tf.not_equal(tf.reduce_sum(index1), 0),
                tf.math.divide(
                    g * index1, (tf.reduce_sum(g * index1) / tf.reduce_sum(index1))
                ),
                tf.zeros_like(g),
            )
            w2 = tf.where(
                tf.not_equal(tf.reduce_sum(index2), 0),
                tf.math.divide(
                    g * index2, (tf.reduce_sum(g * index2) / tf.reduce_sum(index2))
                ),
                tf.zeros_like(g),
            )
            w = w1 + w2
            return w
        else:
            return tf.ones(g.shape, dtype=tf.float32)

    def get_train_set_data(self, class_id):

        images = []
        train_x = self.labeled_train_set[0]
        train_y = self.labeled_train_set[1]
        for i in range(len(train_x)):
            if train_y[i] == class_id:
                images.append(train_x[i])
        return images

    def get_data_size(self):
        logging.info(
            f"self.labeled_train_set is None :{self.labeled_train_set is None}"
        )
        logging.info(
            f"self.unlabeled_train_set is None :{self.unlabeled_train_set is None}"
        )
        data_size = len(self.labeled_train_set[0])
        logging.info(f"data size: {data_size}")
        return data_size

    def _reduce_exemplar_set(self, m):
        for i in range(len(self.exemplar_set)):
            old_exemplar_data = self.exemplar_set[i][0][:m]
            old_exemplar_label = self.exemplar_set[i][1][:m]
            self.exemplar_set[i] = (old_exemplar_data, old_exemplar_label)

    def _construct_exemplar_set(self, images, label, m):
        class_mean, fe_outpu = self.compute_class_mean(images)
        exemplar = []
        labels = []
        now_class_mean = np.zeros((1, 512))
        for i in range(m):
            x = class_mean - (now_class_mean + fe_outpu) / (i + 1)
            x = np.linalg.norm(x)
            index = np.argmin(x)
            now_class_mean += fe_outpu[index]
            exemplar.append(images[index])
            labels.append(label)
        self.exemplar_set.append((exemplar, labels))

    def compute_class_mean(self, images):
        images_data = tf.data.Dataset.from_tensor_slices(images).batch(self.batch_size)
        fe_output = self.feature_extractor.predict(images_data)
        fe_output = tf.nn.l2_normalize(fe_output).numpy()
        class_mean = tf.reduce_mean(fe_output, axis=0)
        return class_mean, fe_output

    def proto_grad(self):
        if self.need_update == False:
            return None
        self.need_update = False
        cri_loss = keras.losses.SparseCategoricalCrossentropy()
        proto = []
        proto_grad = []
        logging.info(f"self. current class is {self.current_classes}")
        for i in self.current_classes:
            images = self.get_train_set_data(i)
            class_mean, fe_output = self.compute_class_mean(images)
            dis = np.linalg.norm(class_mean - fe_output, axis=1)
            pro_index = np.argmin(dis)
            proto.append(images[pro_index])

        for i in range(len(proto)):
            data = proto[i]
            data = tf.cast(tf.expand_dims(data, axis=0), tf.float32)
            label = self.current_classes[i]
            label = tf.constant([label])
            target = get_one_hot(label, self.num_classes)
            logging.info(
                f"proto_grad target shape is {target.shape} and num_classes is {self.num_classes}"
            )
            proto_fe = resnet10()
            proto_fe.build(input_shape=(None, 32, 32, 3))
            proto_fe.call(keras.Input(shape=(32, 32, 3)))
            proto_fe.set_weights(self.feature_extractor.get_weights())
            proto_clf = copy.deepcopy(self.classifier)
            proto_param = proto_fe.trainable_variables
            proto_param.extend(proto_clf.trainable_variables)
            with tf.GradientTape() as tape:
                outputs = self.encode_model(data)
                loss_cls = cri_loss(label, outputs)
            dy_dx = tape.gradient(loss_cls, self.encode_model.trainable_variables)
            original_dy_dx = [tf.identity(grad) for grad in dy_dx]
            proto_grad.append(original_dy_dx)
        return proto_grad

    def evaluate(self):
        logging.info("evaluate")
        total_num = 0
        total_correct = 0
        for x, y in self.train_loader:
            logits = self.model_call(x, training=False)
            pred = tf.argmax(logits, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)
            pred = tf.reshape(pred, y.shape)
            logging.info(pred)
            y = tf.cast(y, dtype=tf.int32)
            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)
            total_num += x.shape[0]
            total_correct += int(correct)
        acc = total_correct / total_num
        del total_correct
        logging.info(f"finsih task {self.old_task_id} evaluate, acc: {acc}")
        return acc
