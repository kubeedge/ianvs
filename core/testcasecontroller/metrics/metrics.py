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

from sedna.common.class_factory import ClassFactory, ClassType

from core.common.constant import SystemMetricKind
from core.common.utils import load_module


def data_transfer_count_ratio(system_metric_info: dict):
    """
    compute data transfer count ratio

    Parameters
    ----------
    system_metric_info: dict
        information needed to compute system metrics.

    Returns
    -------
    float
        e.g.: 0.92

    """

    info = system_metric_info.get(SystemMetricKind.DATA_TRANSFER_COUNT_RATIO.value)
    inference_num = 0
    transfer_num = 0
    for inference_data, transfer_data in info:
        inference_num += len(inference_data)
        transfer_num += len(transfer_data)
    return float(transfer_num) / inference_num


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
        load_module(url)
        try:
            metric_func = ClassFactory.get_cls(type_name=ClassType.GENERAL, t_cls_name=name)
            return name, metric_func
        except Exception as err:
            raise Exception(f"get metric func(url={url}) failed, error: {err}.") from err

    return name, getattr(sys.modules[__name__], name)
