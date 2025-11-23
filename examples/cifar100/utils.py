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

import tensorflow as tf
import numpy as np
import os


def process_cifar100():
    if not os.path.exists("/home/wyd/ianvs/project/data/cifar100"):
        os.makedirs("/home/wyd/ianvs/project/data/cifar100")
    train_txt = "/home/wyd/ianvs/project/data/cifar100/cifar100_train.txt"
    with open(train_txt, "w") as f:
        pass
    test_txt = "/home/wyd/ianvs/project/data/cifar100/cifar100_test.txt"
    with open(test_txt, "w") as f:
        pass
    # load CIFAR-100 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    print(y_test.shape)

    # change label to class index
    class_labels = np.unique(y_train)  #  get all class
    train_class_dict = {label: [] for label in class_labels}
    test_class_dict = {label: [] for label in class_labels}
    # organize training data by category
    for img, label in zip(x_train, y_train):
        train_class_dict[label[0]].append(img)
    # organize testing data by category
    for img, label in zip(x_test, y_test):
        # test_class_dict[label[0]].append(img)
        test_class_dict[label[0]].append(img)
    # save training data to local file
    for label, imgs in train_class_dict.items():
        data = np.array(imgs)
        print(data.shape)
        np.save(
            f"/home/wyd/ianvs/project/data/cifar100/cifar100_train_index_{label}.npy",
            data,
        )
        with open(train_txt, "a") as f:
            f.write(
                f"/home/wyd/ianvs/project/data/cifar100/cifar100_train_index_{label}.npy\t{label}\n"
            )
    #  save test data to local file
    for label, imgs in test_class_dict.items():
        np.save(
            f"/home/wyd/ianvs/project/data/cifar100/cifar100_test_index_{label}.npy",
            np.array(imgs),
        )
        with open(test_txt, "a") as f:
            f.write(
                f"/home/wyd/ianvs/project/data/cifar100/cifar100_test_index_{label}.npy\t{label}\n"
            )
    print(f"CIFAR-100 have saved as ianvs format")


if __name__ == "__main__":
    process_cifar100()
=======
version https://git-lfs.github.com/spec/v1
oid sha256:5bc234ec1f41e021ed1f4ced9dec7252ece913eacea22a08203e69d67b3f7c82
size 2650
>>>>>>> 9676c3e (ya toh aar ya toh par)
