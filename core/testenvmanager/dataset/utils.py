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
import numpy as np

from sedna.datasources import BaseDataSource


def read_data_from_file_to_npy( files: BaseDataSource):
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
    # print(files.x, files.y)
    for i, file in enumerate(files.x):
        x = np.load(file)
        # print(x.shape)
        # print(type(files.y[i]))
        y = np.full((x.shape[0], 1), (files.y[i]).astype(np.int32))
        x_train.append(x)
        y_train.append(y)
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    print(x_train.shape, y_train.shape)
    return x_train, y_train


def partition_data(datasets, client_number, data_partition ='iid'):
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
    print(data_partition)
    if data_partition == 'iid':
        x_data = datasets[0]
        y_data = datasets[1]
        indices = np.arange(len(x_data))
        np.random.shuffle(indices)
        data = []
        for i in range(client_number):
            start = i * len(x_data) // client_number
            end = (i + 1) * len(x_data) // client_number
            data.append([x_data[indices[start:end]], y_data[indices[start:end]]])
        return data
    elif data_partition == 'non-iid':
        pass
    else:
        raise ValueError("paritiion_methods must be 'iid' or 'non-iid'")