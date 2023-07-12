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

"""Base Metrics"""

import sys
import numpy as np
from sedna.common.class_factory import ClassFactory, ClassType

from core.common.constant import SystemMetricType
from core.common.utils import load_module


def samples_transfer_ratio_func(system_metric_info: dict):
    """
    compute samples transfer ratio:
        ratio = nums of all inference samples / nums of all transfer samples

    Parameters
    ----------
    system_metric_info: dict
        information needed to compute system metrics.

    Returns
    -------
    float
        e.g.: 0.92

    """

    info = system_metric_info.get(SystemMetricType.SAMPLES_TRANSFER_RATIO.value)
    inference_num = 0
    transfer_num = 0
    for inference_data, transfer_data in info:
        inference_num += len(inference_data)
        transfer_num += len(transfer_data)
    return round(float(transfer_num) / (inference_num + 1), 4)

def compute(key, matrix):
    """
    compute BWT and FWT
    """
    # pylint: disable=C0103
    length = len(matrix)
    accuracy = 0.0
    BWT_score = 0.0
    FWT_score = 0.0
    flag = True
    for i in range(length):
        if len(matrix[i]) != length-1:
            flag = False
            break
    if flag is False:
        BWT_score = np.nan
        FWT_score = np.nan
        return BWT_score, FWT_score

    for i in range(length-1):
        accuracy += matrix[length-1][i]['accuracy']
        BWT_score += matrix[length-1][i]['accuracy'] - matrix[i+1][i]['accuracy']
    for i in range(0,length-1):
        FWT_score += matrix[i][i]['accuracy'] - matrix[0][i]['accuracy']
    accuracy = accuracy/(length)
    BWT_score = BWT_score/(length-1)
    FWT_score = FWT_score/(length-1)
    #print(f"{key} accuracy: ", accuracy)
    print(f"{key} BWT_score: ", BWT_score)
    print(f"{key} FWT_score: ", FWT_score)
    my_matrix = []
    for i in range(length-1):
        my_matrix.append([])
    for i in range(length-1):
        for j in range(length-1):
            my_matrix[i].append(matrix[i+1][j]['accuracy'])
    #self.draw_picture(key,my_matrix)
    return my_matrix, BWT_score, FWT_score

def bwt_func(system_metric_info: dict):
    """
    compute BWT
    """
    # pylint: disable=C0103
    # pylint: disable=W0632
    info = system_metric_info.get(SystemMetricType.Matrix.value)
    _, BWT_score, _ = compute("all", info["all"])
    return BWT_score

def fwt_func(system_metric_info: dict):
    """
    compute FWT
    """
    # pylint: disable=C0103
    # pylint: disable=W0632
    info = system_metric_info.get(SystemMetricType.Matrix.value)
    _, _, FWT_score = compute("all", info["all"])
    return FWT_score

def matrix_func(system_metric_info: dict):
    """
    compute FWT
    """
    # pylint: disable=C0103
    # pylint: disable=W0632
    info = system_metric_info.get(SystemMetricType.Matrix.value)
    my_dict = {}
    for key in info.keys():
        my_matrix, _, _ = compute(key, info[key])
        my_dict[key] = my_matrix
    return my_dict

def task_avg_acc_func(system_metric_info: dict):
    """
    compute Task_Avg_Acc
    """
    info = system_metric_info.get(SystemMetricType.Task_Avg_Acc.value)
    return info["accuracy"]

def get_metric_func(metric_dict: dict):
    """
    get metric func by metric info

    Parameters:
    ----------
    metric_dict: dict
        metric info, e.g.: {"name": "f1_score", "url": "/metrics/f1_score.py"}

    Returns:
    -------
    name: string
        metric name
    metric_func: function
    """

    name = metric_dict.get("name")
    url = metric_dict.get("url")
    if url:
        try:
            load_module(url)
            metric_func = ClassFactory.get_cls(type_name=ClassType.GENERAL, t_cls_name=name)
            return name, metric_func
        except Exception as err:
            raise RuntimeError(f"get metric func(url={url}) failed, error: {err}.") from err

    return name, getattr(sys.modules[__name__], str.lower(name) + "_func")
