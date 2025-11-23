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
import copy
import logging
import tensorflow as tf
import keras
import numpy as np
from model import resnet10
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
        self.warm_up_round = 1
        self.accept_threshold = 0.85
        self.old_task_id = -1

        self.classifier = None
        self.feature_extractor = self._build_feature_extractor()
        dataset_name = "cifar100"
        self.data_preprocessor = Dataset_Preprocessor(
            dataset_name, Weak_Augment(dataset_name), RandAugment(dataset_name)
        )

        self.observed_classes = []
        self.class_mapping = {}
        self.class_per_round = []
        self.x_exemplars = []
        self.y_exemplars = []
        self.num_meta_round = 5
        self.beta = 0.1
        self.num_rounds = 100
        
        print(f"self epoch is {self.epochs}")

    def _build_feature_extractor(self):
        self.global_model = resnet10(is_combined=True)
        self.global_model.build(input_shape=(None, 32, 32, 3))
        self.global_model.call(keras.Input(shape=(32, 32, 3)))
        feature_extractor = resnet10(is_combined=True)
        feature_extractor.build(input_shape=(None, 32, 32, 3))
        feature_extractor.call(keras.Input(shape=(32, 32, 3)))
        return feature_extractor

    def _build_classifier(self):
        logging.info(f"build classifier with classes {len(self.class_mapping)}")
        if self.classifier != None:
            new_classifier = keras.Sequential(
                [
                    keras.layers.Dense(
                        len(self.class_mapping), kernel_initializer="lecun_normal"
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
                        len(self.class_mapping), kernel_initializer="lecun_normal"
                    )
                ]
            )
            self.classifier.build(
                input_shape=(None, self.feature_extractor.layers[-2].output_shape[-1])
            )


    def get_weights(self):
        return self.feature_extractor.get_weights()

    def set_weights(self, weights):
        self.feature_extractor.set_weights(weights)
        self.global_model.set_weights(weights)

    def get_data_size(self):
        data_size = len(self.labeled_train_set[0]) + len(self.unlabeled_train_set[0])
        logging.info(f"data size: {data_size}")
        return data_size 
    
    def model_call(self, x, training=False):
        x = self.feature_extractor(x, training=training)
        x = self.classifier(x, training=training)
        return x

    def _build_class_mapping(self):
        y_train = self.labeled_train_set[1]
        y = np.unique(y_train)
        logging.info(f'build class mapping, y is {y}')  
        for i in y:
            if not i in self.class_mapping.keys():
                self.class_mapping[i] = len(self.class_mapping)
        self.class_per_round.append([self.class_mapping[i] for i in y])
        logging.info(f'build class mapping, class mapping is {self.class_mapping} and class per round is {self.class_per_round}')
        
    def _mix_with_exemplar(self):
        x_train, y_train = self.labeled_train_set
        if len(self.x_exemplars) == 0:
            return
        x_train = np.concatenate([x_train, np.array(self.x_exemplars)], axis=0)
        y_train = np.concatenate([y_train, np.array(self.y_exemplars)], axis=0)
        self.labeled_train_set = (x_train, y_train)

    def get_train_loader(self):
        label_train_loader = self.data_preprocessor.preprocess_labeled_dataset(
            self.labeled_train_set[0], self.labeled_train_set[1], self.batch_size
        )
        un_label_train_loader = None
        if len(self.unlabeled_train_set[0]) > 0:
            un_label_train_loader = self.data_preprocessor.preprocess_unlabeled_dataset(
                self.unlabeled_train_set[0], self.unlabeled_train_set[1], self.batch_size
            )
        return label_train_loader, un_label_train_loader

    def before_train(self, task_id, round, train_data, task_size):
        if self.task_size is None:
            self.task_size = task_size
        self.labeled_train_set = (train_data["label_x"], train_data["label_y"])
        self.unlabeled_train_set = (
            train_data["unlabel_x"],
            train_data["unlabel_y"],
        )
        self._build_class_mapping()
        self._build_classifier()
        if task_id > 0:
            self._mix_with_exemplar()
        self.feature_extractor.initialize_alpha()
        self.labeled_train_loader, self.unlabeled_train_loader = self.get_train_loader()

    def train(self, task_id, round):
        optimizer = keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9, weight_decay=0.0001)
        all_parameter = []
        all_parameter.extend(self.feature_extractor.trainable_variables)
        all_parameter.extend(self.classifier.trainable_variables)
        
        for epoch in range(self.epochs):
            for x, y in self.labeled_train_loader:
                y = np.array([self.class_mapping[i] for i in y.numpy()])
                tasks = self._split_tasks(x, y)
                base_model_weights = self.feature_extractor.get_weights()
                meta_model_weights = []
                for task_x, task_y in tasks:
                    self.feature_extractor.set_weights(base_model_weights)
                    for _ in range(self.num_meta_round):
                        with tf.GradientTape() as tape:
                            base_loss = self._loss(task_x, task_y)
                            l2_loss = self._loss_l2(self.global_model)
                            loss = base_loss + l2_loss*0.1
                        grads = tape.gradient(loss, all_parameter)
                        optimizer.apply_gradients(zip(grads, all_parameter))
                    meta_model_weights.append(self.feature_extractor.get_weights())
                logging.info(f'Round{round} task{task_id} epoch{epoch} loss is {loss} ')
                self._merge_models(round, base_model_weights, meta_model_weights)
        
        self.feature_extractor.merge_to_local_model()
        self.store_exemplars(task_id)
    
    def evaluate(self):
        total_num = 0
        total_correct = 0
        for x,y in self.labeled_train_loader:
            logits = self.classifier(self.feature_extractor(x, training=False))
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)
            pred = tf.reshape(pred, y.shape)
            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_num += x.shape[0]
            total_correct += int(correct)

        acc = total_correct / total_num
        del total_correct, total_num
        return acc
    
    def _loss(self,x ,y):
        feature = self.feature_extractor(x)
        prediction = self.classifier(feature)
        loss = keras.losses.categorical_crossentropy(tf.one_hot(y, len(self.class_mapping)), prediction, from_logits=True)
        return tf.reduce_mean(loss)
    
    def _loss_l2(self, global_model):
        return 0.0
    
    def unsupervised_loss(self, sux, wux):
        return 0.0
    
    def _merge_models(self, round, base_model_weights, meta_model_weights):
        eta = np.exp(-self.beta * (round + 1 ) / self.num_rounds)
        merged_meta_parameters = [
            np.average(
                [meta_model_weights[i][j] for i in range(len(meta_model_weights))], axis=0
            )for j in range(len(meta_model_weights[0]))
            
        ]
        self.feature_extractor.set_weights([eta * l_meta + (1-eta) * l_base for l_base, l_meta in zip(base_model_weights, merged_meta_parameters)])
        
    def _split_tasks(self, x, y):
        tasks = []
        for classes in self.class_per_round:
            task = None
            for cl in classes:
                x_cl = x[y == cl]
                y_cl = y[y == cl]
                if task is None:
                    task = (x_cl, y_cl)
                else:
                    task = (np.concatenate([task[0], x_cl], axis=0),
                            np.concatenate([task[1], y_cl], axis=0))
            if len(task[0]) > 0:
                self.random_shuffle(task[0],task[1])
                tasks.append(task)
        return tasks

    def random_shuffle(self, x, y):
        p = np.random.permutation(len(x))
        return x[p], y[p]

    def store_exemplars(self, task):
        x = self.labeled_train_set[0]
        y = self.labeled_train_set[1]
        logging.info(f'Storing exemplars..')
        new_classes = self.class_per_round[-1]
        model_classes = np.concatenate(self.class_per_round).tolist()
        old_classes = model_classes[:(-len(new_classes))]
        exemplars_per_class = int(self.memory_size / (len(new_classes) + len(old_classes)))
        
        if task > 0 :
            labels = np.array(self.y_exemplars)
            new_x_exemplars = []
            new_y_exemplars = []
            for cl in old_classes:
                cl_x = np.array(self.x_exemplars)[labels == cl]
                cl_y = np.array(self.y_exemplars)[labels == cl]
                new_x_exemplars.extend(cl_x[:exemplars_per_class])
                new_y_exemplars.extend(cl_y[:exemplars_per_class])
            self.x_exemplars = new_x_exemplars
            self.y_exemplars = new_y_exemplars
        
        for cl in new_classes:
            logging.info(f'Processing class {cl} and y is {y.shape}')
            cl_x = x[y == cl]
            cl_y = y[y == cl]
            
            cl_feat = self.feature_extractor(cl_x)
            cl_mean = tf.reduce_mean(cl_feat, axis=0)
            
            diff = tf.abs(cl_feat - cl_mean)
            distance = [float(tf.reduce_sum(dis).numpy()) for dis in diff]
            
            sorted_index = np.argsort(distance).tolist()
            if len(cl_x) > exemplars_per_class:
                sorted_index = sorted_index[:exemplars_per_class]
            self.x_exemplars.extend(cl_x[sorted_index])
            self.y_exemplars.extend(cl_y[sorted_index])
    
    def predict(self, x):
        mean = np.array((0.5071, 0.4867, 0.4408), np.float32).reshape(1, 1, -1)
        std = np.array((0.2675, 0.2565, 0.2761), np.float32).reshape(1, 1, -1)
        x = (tf.cast(x, dtype=tf.float32) / 255.0 - mean) / std
        pred = self.classifier(self.feature_extractor(x, training=False))
        prob = tf.nn.softmax(pred, axis=1)
        pred = tf.argmax(prob, axis=1)
        pred = tf.cast(pred, dtype=tf.int32)
        return pred 
=======
version https://git-lfs.github.com/spec/v1
oid sha256:8e31ed905eac8a8952f76949519271cb4ba321391e8c4cd430bcdeea373a268e
size 12529
>>>>>>> 9676c3e (ya toh aar ya toh par)
