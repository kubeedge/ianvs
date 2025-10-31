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

import sys
import os
from sedna.core.federated_learning import FederatedLearning
from sedna.datasources import TxtDataParse
import numpy as np
from basemodel import Estimator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from config import TRAIN_INDEX_FILE


def read_data_from_file_to_npy(files):
    """
    read data from file to numpy array

    Parameters
    ---------
    files: list
        the address url of data file.

    Returns
    -------
    list
        data in numpy array.

    """
    x_train = []
    y_train = []
    for i, file in enumerate(files.x):
        x = np.load(file)
        y = np.full((x.shape[0], 1), (files.y[i]).astype(np.int32))
        x_train.append(x)
        y_train.append(y)
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    print(x_train.shape, y_train.shape)
    return x_train, y_train


def main():
    train_file = TRAIN_INDEX_FILE
    train_data = TxtDataParse(data_type="train")
    train_data.parse(train_file)
    train_data = read_data_from_file_to_npy(train_data)
    epochs = 3
    batch_size = 128
    fl_job = FederatedLearning(estimator=Estimator(), aggregation="FedAvg")
    fl_job.train(train_data=train_data, epochs=epochs, batch_size=batch_size)


if __name__ == "__main__":
    main()
