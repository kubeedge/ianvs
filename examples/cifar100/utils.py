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
from config import TRAIN_DIR, TEST_DIR, TRAIN_INDEX_FILE, TEST_INDEX_FILE

def process_cifar100():
    # create train and test directories if they don't exist
    if not os.path.exists(TRAIN_DIR):
        os.makedirs(TRAIN_DIR)
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)

    train_txt = TRAIN_INDEX_FILE
    with open(train_txt, "w") as f:
        pass
    test_txt = TEST_INDEX_FILE
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
            os.path.join(TRAIN_DIR, f"cifar100_train_index_{label}.npy"),
            data,
        )
        with open(train_txt, "a") as f:
            f.write(
                f"{os.path.join(TRAIN_DIR, f'cifar100_train_index_{label}.npy')}\t{label}\n"
            )

    # save test data to local file
    for label, imgs in test_class_dict.items():
        np.save(
            os.path.join(TEST_DIR, f"cifar100_test_index_{label}.npy"),
            np.array(imgs),
        )
        with open(test_txt, "a") as f:
            f.write(
                f"{os.path.join(TEST_DIR, f'cifar100_test_index_{label}.npy')}\t{label}\n"
            )
    print(f"CIFAR-100 have saved as ianvs format")

if __name__ == "__main__":
    process_cifar100()
