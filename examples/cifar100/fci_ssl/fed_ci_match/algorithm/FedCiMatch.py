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

import logging 
import tensorflow as tf
import keras
import numpy as np
from model import resnet10
from agumentation import *
from data_prepocessor import *

def get_one_hot(target, num_classes):
    # print(f'in get  one hot, target shape is {target.shape}')
    y = tf.one_hot(target, depth=num_classes)
    # print(f'in get  one hot, y shape is {y.shape}')
    if len(y.shape) == 3:
        y = tf.squeeze(y, axis=1)
    # print(f'in get  one hot, after tf.squeeze y shape is {y.shape}')
    return y

class FedCiMatch:
    
    def __init__(self, num_classes, batch_size,  epochs, learning_rate, memory_size) -> None:
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
        self.feature_extractor = self.build_feature_extractor()
        
        self.fe_weights_length = 0      
        self.labeled_train_loader = None 
        self.unlabeled_train_loader = None
        self.labeled_train_set = None
        self.unlabeled_train_set = None 
        dataset_name = 'cifar100'
        self.data_preprocessor = Dataset_Preprocessor(dataset_name, Weak_Augment(dataset_name), RandAugment(dataset_name))
        self.last_classes = None
        self.current_classes =None
        self.learned_classes = []
        self.learned_classes_num = 0
        self.exemplar_set = []
        print(f'self epoch is {self.epochs}')
        
    def build_feature_extractor(self):
        feature_extractor = resnet10()
        feature_extractor.build(input_shape=(None, 32, 32, 3))
        feature_extractor.call(keras.Input(shape=(32, 32, 3)))
        return feature_extractor
        
    def build_classifier(self):
        if self.classifier != None:
            new_classifier = keras.Sequential([
                # tf.keras.Input(shape=(None, self.feature_extractor.layers[-2].output_shape[-1])),
                keras.layers.Dense(self.num_classes, kernel_initializer='lecun_normal')
            ])
            new_classifier.build(input_shape=(None, self.feature_extractor.layers[-2].output_shape[-1]))
            new_weights = new_classifier.get_weights()
            old_weights = self.classifier.get_weights()
            # 复制旧参数
            # weight
            new_weights[0][0:old_weights[0].shape[0], 0:old_weights[0].shape[1]] = old_weights[0]
            # bias
            new_weights[1][0:old_weights[1].shape[0]] = old_weights[1]
            new_classifier.set_weights(new_weights)
            self.classifier = new_classifier
        else:
            logging.info(f'input shape is {self.feature_extractor.layers[-2].output_shape[-1]}')
            self.classifier = keras.Sequential([
                # tf.keras.Input(shape=(None, self.feature_extractor.layers[-2].output_shape[-1])),
                keras.layers.Dense(self.num_classes, kernel_initializer='lecun_normal')
            ])
            self.classifier.build(input_shape=(None, self.feature_extractor.layers[-2].output_shape[-1]))
            
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
        fe_weights = weights[:self.fe_weights_length]
        clf_weights = weights[self.fe_weights_length:]
        self.feature_extractor.set_weights(fe_weights)
        self.classifier.set_weights(clf_weights)

    def model_call(self, x, training=False):
        x = self.feature_extractor(x, training=training)
        x = self.classifier(x, training=training)
        # x = tf.nn.softmax(x)
        return x
    
    def before_train(self, task_id, round, train_data,task_size):
        if self.task_size is None:
            self.task_size = task_size
        is_new_task = task_id != self.old_task_id
        if  is_new_task:
            self.old_task_id = task_id
            self.num_classes = self.task_size * (task_id + 1)
            logging.info(f'num_classes: {self.num_classes}')
            if self.current_classes is not None:
                self.last_classes = self.current_classes
            self.build_exemplar(is_new_task)
            self.build_classifier()
            self.current_classes = np.unique(train_data['label_y']).tolist()
            logging.info(f'current_classes: {self.current_classes}')
            
            self.labeled_train_set = (train_data['label_x'], train_data['label_y'])
            self.unlabeled_train_set= (train_data['unlabel_x'], train_data['unlabel_y'])
            logging.info(f'self.labeled_train_set is None :{self.labeled_train_set is None}')
            logging.info(f'self.unlabeled_train_set is None :{self.unlabeled_train_set is None}')
        self.labeled_train_loader, self.unlabeled_train_loader = self.get_train_loader()
        
    def get_data_size(self):
        logging.info(f'self.labeled_train_set is None :{self.labeled_train_set is None}')
        logging.info(f'self.unlabeled_train_set is None :{self.unlabeled_train_set is None}')
        data_size = len(self.labeled_train_set[0]) + len(self.unlabeled_train_set[0]) 
        logging.info(f"data size: {data_size}")
        return data_size  
    
    def get_train_loader(self):
        train_x = self.labeled_train_set[0]
        train_y = self.labeled_train_set[1]
        print(f'train_x shape: {train_x.shape} and train_y shape: {train_y.shape}')
        if len(self.exemplar_set) != 0:
            for exm_set in self.exemplar_set:
                # print('in get train loader' , exm_set[0].shape)
                train_x = np.concatenate((train_x, exm_set[0]), axis=0)
                label = np.array(exm_set[1])
                label = label.reshape(-1, 1)
                train_y = np.concatenate((train_y, label), axis=0)
            print(f'unlabel_x shape: {self.unlabeled_train_set[0].shape} and unlabel_y shape: {self.unlabeled_train_set[1].shape}')
        
        label_data_loader = self.data_preprocessor.preprocess_labeled_dataset(train_x, train_y, self.batch_size)
        unlabel_data_loader = None
        if len(self.unlabeled_train_set[0]) > 0:
            unlabel_data_loader = self.data_preprocessor.preprocess_unlabeled_dataset(self.unlabeled_train_set[0], self.unlabeled_train_set[1], self.batch_size) 
        return label_data_loader, unlabel_data_loader
            
    
    def build_exemplar(self, is_new_task):
        if is_new_task and self.last_classes is not None:
            self.learned_classes.extend(self.last_classes)
            self.learned_classes_num += len(self.learned_classes)
            m = int(self.memory_size / self.num_classes)
            self.reduce_exemplar_set(m)
            for cls in self.last_classes:
                images = self.get_train_data(cls)
                self.construct_exemplar_set(images, cls, m)
    
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
        
    def construct_exemplar_set(self, images,class_id, m):
        exemplar_data = []
        exemplar_label = []
        class_mean, fe_ouput = self.compute_exemplar_mean(images)
        now_class_mean = np.zeros((1, fe_ouput.shape[1]))
        for i in range(m):
            x = class_mean - (now_class_mean + fe_ouput)/(i+1)
            x = np.linalg.norm(x)
            index = np.argmin(x)
            now_class_mean += fe_ouput[index]
            exemplar_data.append(images[index])
            exemplar_label.append(class_id)
        self.exemplar_set.append((exemplar_data, exemplar_label))
    
    def compute_exemplar_mean(self, images):
       images_data = tf.data.Dataset.from_tensor_slices(images).batch(self.batch_size).map(lambda x: tf.cast(x, dtype=tf.float32) / 255.)
       fe_output = self.feature_extractor.predict(images_data)
       print('fe_output shape:', fe_output.shape)
       fe_output = tf.nn.l2_normalize(fe_output).numpy()
       class_mean = np.mean(fe_output, axis=0)
       return class_mean, fe_output
   
    def train(self, round):
        
        optimizer = keras.optimizers.SGD(learning_rate=self.learning_rate, weight_decay=0.00001)
        feature_extractor_params = self.feature_extractor.trainable_variables
        classifier_params = self.classifier.trainable_variables
        all_params = []
        all_params.extend(feature_extractor_params)
        all_params.extend(classifier_params)
        # all_params = []
        # all_params.extend(self.feature_extractor.trainable_variables)
        # all_params.extend(self.classifier.trainable_variables)

        for epoch in range(self.epochs):
            # for (labeled_data, unlabeled_data) in zip(self.labeled_train_loader, self.unlabeled_train_loader):
            for step, (labeled_x, labeled_y) in enumerate(self.labeled_train_loader):
                # print(labeled_data.shape)
                # labeled_x, labeled_y = labeled_data
                # unlabeled_x, weak_unlabeled_x, strong_unlabeled_x, unlabeled_y = unlabeled_data
                with tf.GradientTape() as tape:
                    input = self.feature_extractor(inputs=labeled_x,training=True)
                    y_pred = self.classifier(inputs=input, training=True)
                    # target = get_one_hot(labeled_y, self.num_classes)
                    label_pred = tf.argmax(y_pred, axis=1)
                    label_pred = tf.cast(label_pred, dtype=tf.int32)
                    label_pred = tf.reshape(label_pred, labeled_y.shape)
                    # logging.info(f"{label_pred.numpy()}")
                    correct = tf.reduce_sum(tf.cast(tf.equal(label_pred, labeled_y), dtype=tf.int32))
                    loss = tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(labeled_y, y_pred, from_logits=True))
                    # if round > self.warm_up_round: 
                    #     unsupervised_loss = self.unsupervised_loss(weak_unlabeled_x, strong_unlabeled_x, unlabeled_x)        
                    #     loss = 0.5 * supervised_loss + 0.5 * unsupervised_loss
                logging.info(f"epoch {epoch} step {step} loss: {loss} correct {correct} and total {labeled_x.shape[0]}")
                grads = tape.gradient(loss, all_params)
                optimizer.apply_gradients(zip(grads, all_params))
    
    def supervised_loss(self, x, y):
        x = self.feature_extractor(x,training=True)
        y_pred = tf.nn.softmax(self.classifier(x, training=True))
        # y_pred = tf.nn.softmax( self.model_call(x, training=True)) 
        target = get_one_hot(y, self.num_classes)
        # logits = y_pred
        #     # prob = tf.nn.softmax(logits, axis=1)
        # pred = tf.argmax(logits, axis=1)
        # pred = tf.cast(pred, dtype=tf.int32)
        # pred = tf.reshape(pred, y.shape)
        
        # labels = tf.cast(y, dtype=tf.int32)
        # correct = tf.cast(tf.equal(pred, labels), dtype=tf.int32)
        # correct = tf.reduce_sum(correct)
        # logging.info(f'current class numbers is {self.num_classes} correct is {correct} and acc is {correct/x.shape[0]}')
        loss = tf.reduce_mean(keras.losses.categorical_crossentropy(target, y_pred, from_logits=True) )

        return loss
      

    def unsupervised_loss(self, weak_x, strong_x, x):
        prob_on_wux = tf.nn.softmax(self.model_call(weak_x, training=True))
        pseudo_mask = tf.cast(tf.reduce_max(prob_on_wux, axis=1) > self.accept_threshold, tf.float32)
        pse_uy = tf.one_hot(tf.argmax(prob_on_wux, axis=1), depth=self.num_classes)
        prob_on_sux = tf.nn.softmax(self.model_call(strong_x, training=True))
        loss = keras.losses.categorical_crossentropy(pse_uy, prob_on_sux, from_logits=True)
        loss = tf.reduce_mean(loss * pseudo_mask)
        return loss
    
    def predict(self, x):
        mean = np.array((0.5071, 0.4867, 0.4408), np.float32).reshape(1, 1, -1)
        std = np.array((0.2675, 0.2565, 0.2761), np.float32).reshape(1, 1, -1)
        x = (tf.cast(x, dtype=tf.float32) / 255. - mean) /std
        x = self.feature_extractor(x, training=False)
        y_pred = tf.nn.softmax(self.classifier(x, training=False))
        # logging.info(f"y_pred  is {y_pred}")
        pred = tf.cast(tf.argmax(y_pred, axis=1), tf.int32)
        return pred