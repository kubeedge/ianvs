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

"""Base constant"""

from enum import Enum


class DatasetFormat(Enum):
    """
    File format of inputting dataset.
    Currently, file formats are as follows: txt, csv.
    """
    CSV = "csv"
    TXT = "txt"


class ParadigmKind(Enum):
    """
    Algorithm paradigm kind.
    """
    SINGLE_TASK_LEARNING = "singletasklearning"
    INCREMENTAL_LEARNING = "incrementallearning"


class ModuleKind(Enum):
    """
    Algorithm module kind.
    """
    BASEMODEL = "basemodel"
    HARD_EXAMPLE_MINING = "hard_example_mining"


class SystemMetricKind(Enum):
    """
    System metric kind of ianvs.
    """
    DATA_TRANSFER_COUNT_RATIO = "data_transfer_count_ratio"
