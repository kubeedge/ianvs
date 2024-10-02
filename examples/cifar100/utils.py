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

import tensorflow as tf
import numpy as np
import os


def process_cifar100():
    if not os.path.exists('/home/wyd/ianvs/project/data/cifar100'):
        os.makedirs('/home/wyd/ianvs/project/data/cifar100')
    train_txt = '/home/wyd/ianvs/project/data/cifar100/cifar100_train.txt'
    with open(train_txt, 'w') as f:
        pass
    test_txt = '/home/wyd/ianvs/project/data/cifar100/cifar100_test.txt'
    with open(test_txt, 'w') as f:
        pass
    # 加载CIFAR-100数据集
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    print(y_test.shape)
    # 数据预处理：归一化
    # x_train, x_test = x_train / 255.0, x_test / 255.0

    # 将标签转换为类别索引
    class_labels = np.unique(y_train)  # 获取所有类别
    train_class_dict = {label: [] for label in class_labels}
    test_class_dict = {label: [] for label in class_labels}
    # train_cnt = 0
    # train_file_str = []
    # 按类别组织训练数据
    for img, label in zip(x_train, y_train):
        # print(type(img))
        # print('----')
        train_class_dict[label[0]].append(img)
    # # 按类别组织测试数据
    for img, label in zip(x_test, y_test):
        # test_class_dict[label[0]].append(img)
        test_class_dict[label[0]].append(img)
    # 保存训练数据到本地文件
    for label, imgs in train_class_dict.items():
        data = np.array(imgs)
        print(data.shape)
        np.save(f'/home/wyd/ianvs/project/data/cifar100/cifar100_train_index_{label}.npy',data)
        with open(train_txt, 'a') as f:
            f.write(f'/home/wyd/ianvs/project/data/cifar100/cifar100_train_index_{label}.npy\t{label}\n')
    # 保存测试数据到本地文件
    for label, imgs in test_class_dict.items():
        np.save(f'/home/wyd/ianvs/project/data/cifar100/cifar100_test_index_{label}.npy', np.array(imgs))
        with open(test_txt, 'a') as f:
            f.write(f'/home/wyd/ianvs/project/data/cifar100/cifar100_test_index_{label}.npy\t{label}\n')
    print(f'CIFAR-100 数据集已按类别保存到本地文件。')



if __name__ == '__main__':
    process_cifar100()
