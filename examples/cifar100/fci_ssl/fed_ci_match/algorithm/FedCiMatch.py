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
import logging
import tensorflow as tf
import keras
import numpy as np
from model import resnet10, resnet18
from agumentation import *
from data_prepocessor import *


def get_one_hot(target, num_classes):
    y = tf.one_hot(target, depth=num_classes)
    if len(y.shape) == 3:
        y = tf.squeeze(y, axis=1)
    return y


class FedCiMatch:

    def __init__(
        self, num_classes, batch_size, epochs, learning_rate, memory_size
    ) -> None:
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.task_size = None
        self.warm_up_round = 4
        self.accept_threshold = 0.95
        self.old_task_id = -1

        self.classifier = None
        self.feature_extractor = self.build_feature_extractor()

        self.fe_weights_length = 0
        self.labeled_train_loader = None
        self.unlabeled_train_loader = None
        self.labeled_train_set = None
        self.unlabeled_train_set = None
        dataset_name = "cifar100"
        self.data_preprocessor = Dataset_Preprocessor(
            dataset_name, Weak_Augment(dataset_name), RandAugment(dataset_name)
        )
        self.last_classes = None
        self.current_classes = None
        self.learned_classes = []
        self.learned_classes_num = 0
        self.exemplar_set = []
        self.seen_classes = []
        self.best_old_model = None
        print(f"self epoch is {self.epochs}")

    def build_feature_extractor(self):
        feature_extractor = resnet18()

        feature_extractor.build(input_shape=(None, 32, 32, 3))
        feature_extractor.call(keras.Input(shape=(32, 32, 3)))
        return feature_extractor

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
            self.classifier = keras.Sequential(
                [
                    keras.layers.Dense(
                        self.num_classes, kernel_initializer="lecun_normal"
                    )
                ]
            )
            self.classifier.build(
                input_shape=(None, self.feature_extractor.layers[-2].output_shape[-1])
            )

        logging.info(f"finish ! initialize classifier {self.classifier.summary()}")

    def get_weights(self):
        weights = []
        fe_weights = self.feature_extractor.get_weights()
        self.fe_weights_length = len(fe_weights)
        clf_weights = self.classifier.get_weights()
        weights.extend(fe_weights)
        weights.extend(clf_weights)
        return weights

    def set_weights(self, weights):
        fe_weights = weights[: self.fe_weights_length]
        clf_weights = weights[self.fe_weights_length :]
        self.feature_extractor.set_weights(fe_weights)
        self.classifier.set_weights(clf_weights)

    def model_call(self, x, training=False):
        x = self.feature_extractor(x, training=training)
        x = self.classifier(x, training=training)
        # x = tf.nn.softmax(x)
        return x

    def before_train(self, task_id, round, train_data, task_size):
        if self.task_size is None:
            self.task_size = task_size
        is_new_task = task_id != self.old_task_id
        self.is_new_task = is_new_task
        if is_new_task:
            self.best_old_model = (
                (self.feature_extractor, self.classifier)
                if self.classifier is not None
                else None
            )
            self.is_new_task = True
            self.old_task_id = task_id
            self.num_classes = self.task_size * (task_id + 1)
            logging.info(f"num_classes: {self.num_classes}")
            if self.current_classes is not None:
                self.last_classes = self.current_classes
            # self.build_classifier()
            self.current_classes = np.unique(train_data["label_y"]).tolist()
            logging.info(f"current_classes: {self.current_classes}")

            self.labeled_train_set = (train_data["label_x"], train_data["label_y"])
            self.unlabeled_train_set = (
                train_data["unlabel_x"],
                train_data["unlabel_y"],
            )
            logging.info(
                f"self.labeled_train_set is None :{self.labeled_train_set is None}"
            )
            logging.info(
                f"self.unlabeled_train_set is None :{self.unlabeled_train_set is None}"
            )
        self.labeled_train_loader, self.unlabeled_train_loader = self.get_train_loader()

    def get_data_size(self):
        logging.info(
            f"self.labeled_train_set is None :{self.labeled_train_set is None}"
        )
        logging.info(
            f"self.unlabeled_train_set is None :{self.unlabeled_train_set is None}"
        )
        data_size = len(self.labeled_train_set[0]) + len(self.unlabeled_train_set[0])
        logging.info(f"data size: {data_size}")
        return data_size

    def get_train_loader(self):
        train_x = self.labeled_train_set[0]
        train_y = self.labeled_train_set[1]
        logging.info(
            f"train_x shape: {train_x.shape} and train_y shape: {train_y.shape} and len of exemplar_set: {len(self.exemplar_set)}"
        )
        if len(self.exemplar_set) != 0:
            for exm_set in self.exemplar_set:
                train_x = np.concatenate((train_x, exm_set[0]), axis=0)
                label = np.array(exm_set[1])
                train_y = np.concatenate((train_y, label), axis=0)
            logging.info(
                f"train_x shape: {train_x.shape} and train_y shape: {train_y.shape}"
            )

        logging.info(
            f"train_x shape: {train_x.shape} and train_y shape: {train_y.shape}"
        )
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
        return label_data_loader, unlabel_data_loader

    def build_exemplar(self):
        if self.is_new_task and self.current_classes is not None:
            self.last_classes = self.current_classes
            self.learned_classes.extend(self.last_classes)
            self.learned_classes_num += len(self.learned_classes)
            m = int(self.memory_size / self.num_classes)
            self.reduce_exemplar_set(m)
            for cls in self.last_classes:
                images = self.get_train_data(cls)
                self.construct_exemplar_set(images, cls, m)
            self.is_new_task = False

    def reduce_exemplar_set(self, m):
        for i in range(len(self.exemplar_set)):
            old_exemplar_data = self.exemplar_set[i][0][:m]
            old_exemplar_label = self.exemplar_set[i][1][:m]
            self.exemplar_set[i] = (old_exemplar_data, old_exemplar_label)

    def get_train_data(self, class_id):
        images = []
        train_x = self.labeled_train_set[0]
        train_y = self.labeled_train_set[1]
        for i in range(len(train_x)):
            if train_y[i] == class_id:
                images.append(train_x[i])
        return images

    def construct_exemplar_set(self, images, class_id, m):
        exemplar_data = []
        exemplar_label = []
        class_mean, fe_ouput = self.compute_exemplar_mean(images)
        diff = tf.abs(fe_ouput - class_mean)
        distance = [float(tf.reduce_sum(dis).numpy()) for dis in diff]

        sorted_index = np.argsort(distance).tolist()
        if len(sorted_index) > m:
            sorted_index = sorted_index[:m]
        exemplar_data = [images[i] for i in sorted_index]
        exemplar_label = [class_id] * len(exemplar_data)
        self.exemplar_set.append((exemplar_data, exemplar_label))


    def compute_exemplar_mean(self, images):
        images_data = (
            tf.data.Dataset.from_tensor_slices(images)
            .batch(self.batch_size)
            .map(lambda x: tf.cast(x, dtype=tf.float32) / 255.0)
        )
        fe_output = self.feature_extractor.predict(images_data)
        print("fe_output shape:", fe_output.shape)
        class_mean = tf.reduce_mean(fe_output, axis=0)
        return class_mean, fe_output

    def train(self, round):
        # optimizer = keras.optimizers.SGD(
        #     learning_rate=self.learning_rate, momentum=0.9, weight_decay=0.0001
        # )
        optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate, weight_decay=0.0001
        )
        q = []
        logging.info(f"is new task: {self.is_new_task}")
        if self.is_new_task:
            self.build_classifier()
        all_params = []
        all_params.extend(self.feature_extractor.trainable_variables)
        all_params.extend(self.classifier.trainable_variables)

        for epoch in range(self.epochs):
            # following code is for unsupervised learning
            # for labeled_data, unlabeled_data in zip(
            #     self.labeled_train_loader, self.unlabeled_train_loader
            # ):
            for step, (labeled_x, labeled_y) in enumerate(self.labeled_train_loader):
                with tf.GradientTape() as tape:
                    input = self.feature_extractor(inputs=labeled_x, training=True)
                    y_pred = self.classifier(inputs=input, training=True)
                    label_pred = tf.argmax(y_pred, axis=1)
                    label_pred = tf.cast(label_pred, dtype=tf.int32)
                    label_pred = tf.reshape(label_pred, labeled_y.shape)
                    correct = tf.reduce_sum(
                        tf.cast(tf.equal(label_pred, labeled_y), dtype=tf.int32)
                    )
                    CE_loss = self.supervised_loss(labeled_x, labeled_y)
                    KD_loss = self.distil_loss(labeled_x, labeled_y)
                    supervised_loss = CE_loss

                    # following code is for unsupervised learning
                    # if epoch > self.warm_up_round:
                    #     unsupervised_loss = self.unsupervised_loss(
                    #         weak_unlabeled_x, strong_unlabeled_x, unlabeled_x
                    #     )
                    #     logging.info(f"unsupervised loss: {unsupervised_loss}")
                    #     loss = 0.5 * supervised_loss + 0.5 * unsupervised_loss
                    # else:
                    #     loss = supervised_loss
                    loss = CE_loss + KD_loss
                logging.info(
                    f"epoch {epoch}  loss: {loss}  correct {correct} and total {labeled_x.shape[0]} class is {np.unique(labeled_y)}"
                )
                grads = tape.gradient(loss, all_params)
                optimizer.apply_gradients(zip(grads, all_params))

    def caculate_pre_update(self):
        q = []
        for images, _ in self.labeled_train_loader:
            x = self.feature_extractor(images, training=False)
            x = self.classifier(x, training=False)
            x = tf.nn.sigmoid(x)
            q.append(x)
        logging.info(f"q shape: {len(q)}")
        return q

    def supervised_loss(self, x, y):
        input = x
        input = self.feature_extractor(input, training=True)
        y_pred = self.classifier(input, training=True)
        target = get_one_hot(y, self.num_classes)
        loss = keras.losses.categorical_crossentropy(target, y_pred, from_logits=True)
        logging.info(f"loss shape: {loss.shape}")
        loss = tf.reduce_mean(loss)
        logging.info(f"CE loss: {loss}")

        return loss

    def distil_loss(self, x, y):
        KD_loss = 0

        if len(self.learned_classes) > 0 and self.best_old_model is not None:
            g = self.feature_extractor(x, training=True)
            g = self.classifier(g, training=True)
            og = self.best_old_model[0](x, training=False)
            og = self.best_old_model[1](og, training=False)
            sigmoid_og = tf.nn.sigmoid(og)
            sigmoid_g = tf.nn.sigmoid(g)
            BCELoss = keras.losses.BinaryCrossentropy()
            loss = []
            for y in self.learned_classes:
                if y not in self.current_classes:
                    loss.append(BCELoss(sigmoid_og[:, y], sigmoid_g[:, y]))
            KD_loss = tf.reduce_sum(loss)
        logging.info(f"KD_loss: {KD_loss}")
        return KD_loss

    def unsupervised_loss(self, weak_x, strong_x, x):
        prob_on_wux = tf.nn.softmax(
            self.classifier(
                self.feature_extractor(weak_x, training=True), training=True
            )
        )
        pseudo_mask = tf.cast(
            tf.reduce_max(prob_on_wux, axis=1) > self.accept_threshold, tf.float32
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

    def predict(self, x):
        mean = np.array((0.5071, 0.4867, 0.4408), np.float32).reshape(1, 1, -1)
        std = np.array((0.2675, 0.2565, 0.2761), np.float32).reshape(1, 1, -1)
        x = (tf.cast(x, dtype=tf.float32) / 255.0 - mean) / std
        pred = self.classifier(self.feature_extractor(x, training=False))
        prob = tf.nn.softmax(pred, axis=1)
        pred = tf.argmax(prob, axis=1)
        pred = tf.cast(pred, dtype=tf.int32)
        return pred

    def icarl_predict(self, x):
        mean = np.array((0.5071, 0.4867, 0.4408), np.float32).reshape(1, 1, -1)
        std = np.array((0.2675, 0.2565, 0.2761), np.float32).reshape(1, 1, -1)
        x = (tf.cast(x, dtype=tf.float32) / 255.0 - mean) / std
        bs = x.shape[0]
        print(x.shape)
        exemplar_mean = []
        for exemplar in self.exemplar_set:
            # features = []
            ex, _ = exemplar
            ex = (tf.cast(ex, dtype=tf.float32) / 255.0 - mean) / std
            feature = self.feature_extractor(ex, training=False)
            feature = feature / tf.norm(feature)
            mu_y = tf.reduce_mean(feature, axis=0)
            mu_y = mu_y / tf.norm(mu_y)
            exemplar_mean.append(mu_y)
        means = tf.stack(exemplar_mean)  # shape: (num_classes, feature_shape)
        means = tf.stack([means] * bs)  # shape: (bs, num_classes, feature_shape)
        means = tf.transpose(
            means, perm=[0, 2, 1]
        )  # shape: (bs, feature_shape, num_classes)
        feature = self.feature_extractor(
            x, training=False
        )  # shape  (bs , feature_shape)
        feature = feature / tf.norm(feature)
        feature = tf.expand_dims(feature, axis=2)
        feature = tf.tile(feature, [1, 1, self.num_classes])
        dists = tf.pow((feature - means), 2)
        dists = tf.reduce_sum(dists, axis=1)  # shape: (bs, num_classes)
        preds = tf.argmin(dists, axis=1)  # shape: (bs)
        logging.info(f"preds : {preds}")
        return preds
