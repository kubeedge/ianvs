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


class ParadigmType(Enum):
    """
    Algorithm paradigm type.
    """
    SINGLE_TASK_LEARNING = "singletasklearning"
    INCREMENTAL_LEARNING = "incrementallearning"
    LIFELONG_LEARNING = "lifelonglearning"


class ModuleType(Enum):
    """
    Algorithm module type.
    """
    BASEMODEL = "basemodel"

    # HEM
    HARD_EXAMPLE_MINING = "hard_example_mining"

    # STP
    TASK_DEFINITION = "task_definition"
    TASK_RELATIONSHIP_DISCOVERY = "task_relationship_discovery"
    TASK_ALLOCATION = "task_allocation"
    TASK_REMODELING = "task_remodeling"
    INFERENCE_INTEGRATE = "inference_integrate"

    # KM
    TASK_UPDATE_DECISION = "task_update_decision"

    # UTP
    UNSEEN_TASK_ALLOCATION = "unseen_task_allocation"

    # UTD
    UNSEEN_SAMPLE_RECOGNITION = "unseen_sample_recognition"
    UNSEEN_SAMPLE_RE_RECOGNITION = "unseen_sample_re_recognition"


class SystemMetricType(Enum):
    """
    System metric type of ianvs.
    """
    SAMPLES_TRANSFER_RATIO = "samples_transfer_ratio"


class TestObjectType(Enum):
    """
    Test object type of ianvs.
    """
    ALGORITHMS = "algorithms"
