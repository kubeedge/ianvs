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
        ratio = nums of all transfer samples / nums of all inference samples

    Parameters
    ----------
    system_metric_info: dict
        information needed to compute system metrics.

    Returns
    -------
    float
        e.g.: 0.92
    """
    if not system_metric_info:
        return np.nan

    info = system_metric_info.get(SystemMetricType.SAMPLES_TRANSFER_RATIO.value)
    if not info or not isinstance(info, (list, tuple)):
        return np.nan

    inference_num = 0
    transfer_num = 0
    for item in info:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            inference_data, transfer_data = item
            inference_num += len(inference_data) if hasattr(inference_data, "__len__") else 0
            transfer_num += len(transfer_data) if hasattr(transfer_data, "__len__") else 0

    if inference_num == 0:
        return 0.0 if transfer_num == 0 else np.nan

    return round(float(transfer_num) / float(inference_num), 4)


def compute(key, matrix):
    """
    Compute BWT and FWT scores for a given matrix.
    """
    if not matrix or not isinstance(matrix, list) or len(matrix) <= 1:
        return [], np.nan, np.nan

    length = len(matrix)
    accuracy = 0.0
    bwt_score = 0.0
    fwt_score = 0.0
    flag = True

    for row in matrix:
        if not isinstance(row, list) or len(row) != length - 1:
            flag = False
            break

    if not flag:
        bwt_score = np.nan
        fwt_score = np.nan
        return [], bwt_score, fwt_score

    for i in range(length - 1):
        for j in range(length - 1):
            if "accuracy" in matrix[i + 1][j] and "accuracy" in matrix[i][j]:
                accuracy += matrix[i + 1][j]["accuracy"]
                bwt_score += matrix[i + 1][j]["accuracy"] - matrix[i][j]["accuracy"]

    for i in range(0, length - 1):
        if "accuracy" in matrix[i][i] and "accuracy" in matrix[0][i]:
            fwt_score += matrix[i][i]["accuracy"] - matrix[0][i]["accuracy"]

    accuracy = accuracy / ((length - 1) * (length - 1))
    bwt_score = bwt_score / ((length - 1) * (length - 1))
    fwt_score = fwt_score / (length - 1)

    print(f"{key} BWT_score: {bwt_score}")
    print(f"{key} FWT_score: {fwt_score}")

    my_matrix = []
    for i in range(length - 1):
        my_matrix.append([])
        for j in range(length - 1):
            if "accuracy" in matrix[i + 1][j]:
                my_matrix[i].append(matrix[i + 1][j]["accuracy"])

    return my_matrix, bwt_score, fwt_score


def bwt_func(system_metric_info: dict):
    """
    compute BWT
    """
    # pylint: disable=C0103
    # pylint: disable=W0632
    if not system_metric_info:
        return np.nan
    info = system_metric_info.get(SystemMetricType.MATRIX.value)
    if not info or not isinstance(info, dict) or "all" not in info:
        return np.nan
    _, BWT_score, _ = compute("all", info["all"])
    return BWT_score


def fwt_func(system_metric_info: dict):
    """
    compute FWT
    """
    # pylint: disable=C0103
    # pylint: disable=W0632
    if not system_metric_info:
        return np.nan
    info = system_metric_info.get(SystemMetricType.MATRIX.value)
    if not info or not isinstance(info, dict) or "all" not in info:
        return np.nan
    _, _, FWT_score = compute("all", info["all"])
    return FWT_score


def matrix_func(system_metric_info: dict):
    """
    compute FWT
    """
    # pylint: disable=C0103
    # pylint: disable=W0632
    if not system_metric_info:
        return {}
    info = system_metric_info.get(SystemMetricType.MATRIX.value)
    if not info or not isinstance(info, dict):
        return {}
    my_dict = {}
    for key in info.keys():
        my_matrix, _, _ = compute(key, info[key])
        my_dict[key] = my_matrix
    return my_dict


def task_avg_acc_func(system_metric_info: dict):
    """
    compute task average accuracy
    """
    if not system_metric_info:
        return np.nan
    info = system_metric_info.get(SystemMetricType.TASK_AVG_ACC.value)
    if not info or not isinstance(info, dict) or "accuracy" not in info:
        return np.nan
    return round(info["accuracy"], 3)


def forget_rate_func(system_metric_info: dict):
    """
    compute task forget rate
    """
    if not system_metric_info:
        return np.nan
    info = system_metric_info.get(SystemMetricType.FORGET_RATE.value)
    if info is None or (hasattr(info, "__len__") and len(info) == 0):
        return np.nan
    try:
        forget_rate = np.mean(info)
        if np.isnan(forget_rate):
            return np.nan
        return round(float(forget_rate), 3)
    except Exception:
        return np.nan


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
            metric_func = ClassFactory.get_cls(
                type_name=ClassType.GENERAL, t_cls_name=name
            )
            return name, metric_func
        except Exception as err:
            raise RuntimeError(
                f"get metric func(url={url}) failed, error: {err}."
            ) from err

    func_name = str.lower(name) + "_func" if name else ""
    if func_name and hasattr(sys.modules[__name__], func_name):
        return name, getattr(sys.modules[__name__], func_name)

    raise ValueError(
        f"metric func for '{name}' is not found in built-in metrics "
        f"and no external plugin url was provided."
    )
