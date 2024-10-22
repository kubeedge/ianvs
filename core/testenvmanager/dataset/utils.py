# Copyright 2022 The KubeEdge Authors.
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

""" Dataset utils to read data from file and partition data """
# pylint: disable=W1203
import random
import numpy as np
from sedna.datasources import BaseDataSource
from core.common.log import LOGGER


def read_data_from_file_to_npy(files: BaseDataSource):
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
        x_data = np.load(file)
        y_data = np.full((x_data.shape[0],), (files.y[i]).astype(np.int32))
        x_train.append(x_data)
        y_train.append(y_data)
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    return x_train, y_train


def partition_data(datasets, client_number, data_partition="iid", non_iid_ratio=0.6):
    """
    Partition data into clients.

    Parameters
    ----------
    datasets: list
        The list containing the data and labels (x_data, y_data).
    client_number: int
        The number of clients.
    partition_methods: str
        The partition method, either 'iid' or 'non-iid'.

    Returns
    -------
    list
        A list of data for each client in numpy array format.
    """
    LOGGER.info(data_partition)
    data = []
    if data_partition == "iid":
        x_data = datasets[0]
        y_data = datasets[1]
        indices = np.arange(len(x_data))
        np.random.shuffle(indices)
        for i in range(client_number):
            start = i * len(x_data) // client_number
            end = (i + 1) * len(x_data) // client_number
            data.append([x_data[indices[start:end]], y_data[indices[start:end]]])
    elif data_partition == "non-iid":
        class_num = len(np.unique(datasets[1]))
        x_data = datasets[0]
        y_data = datasets[1]

        for i in range(client_number):
            sample_number = int(class_num * non_iid_ratio)
            current_class = random.sample(range(class_num), sample_number)
            LOGGER.info(f"for client{i} the class is {current_class}")
            indices = np.where(y_data == current_class)[0]
            data.append([x_data[indices], y_data[indices]])
    else:
        raise ValueError("paritiion_methods must be 'iid' or 'non-iid'")
    return data
