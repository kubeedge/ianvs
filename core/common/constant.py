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
    JSON = "json"
    JSONL = "jsonl"
    JSONFORLLM = "jsonforllm"


class ParadigmType(Enum):
    """
    Algorithm paradigm type.
    """

    SINGLE_TASK_LEARNING = "singletasklearning"
    INCREMENTAL_LEARNING = "incrementallearning"
    MULTIEDGE_INFERENCE = "multiedgeinference"
    LIFELONG_LEARNING = "lifelonglearning"
    FEDERATED_LEARNING = "federatedlearning"
    FEDERATED_CLASS_INCREMENTAL_LEARNING = "federatedclassincrementallearning"
    JOINT_INFERENCE = "jointinference"


class ModuleType(Enum):
    """
    Algorithm module type.
    """

    BASEMODEL = "basemodel"

    # JOINT INFERENCE
    EDGEMODEL = "edgemodel"
    CLOUDMODEL = "cloudmodel"
    DRAFTER = "drafter"
    VERIFIER = "verifier"

    # Dataset Preprocessor
    DATA_PROCESSOR = "dataset_processor"

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

    # FL_AGG
    AGGREGATION = "aggregation"


class SystemMetricType(Enum):
    """
    System metric type of ianvs.
    """

    SAMPLES_TRANSFER_RATIO = "samples_transfer_ratio"
    FWT = "FWT"
    BWT = "BWT"
    TASK_AVG_ACC = "task_avg_acc"
    MATRIX = "MATRIX"
    FORGET_RATE = "forget_rate"


class TestObjectType(Enum):
    """
    Test object type of ianvs.
    """

    ALGORITHMS = "algorithms"


class EnvKey(str, Enum):
    """
    Canonical names of the environment variables that ianvs sets for algorithm modules.

    Using these constants instead of inline strings eliminates typos, provides a single
    authoritative reference, and makes the YAML-key → env-var mapping explicit.

    Mapping to the YAML ``algorithm`` config key:
        ``initial_model_url``  →  :attr:`BASE_MODEL_URL`
    """

    # Path to the starting model supplied via ``initial_model_url`` in the algorithm YAML.
    # Algorithm train() implementations should read this to locate the base checkpoint.
    BASE_MODEL_URL = "BASE_MODEL_URL"

    # Output path where the algorithm should write the newly trained model.
    MODEL_URL = "MODEL_URL"

    # Semicolon-separated list of model paths used during evaluation and lifelong learning.
    MODEL_URLS = "MODEL_URLS"

    # Base output directory for the current paradigm step (train / eval / inference).
    OUTPUT_URL = "OUTPUT_URL"

    # Directory where inference result files should be written.
    RESULT_SAVED_URL = "RESULT_SAVED_URL"

    # Directory for per-sample inference result files in lifelong learning.
    INFERENCE_RESULT_DIR = "INFERENCE_RESULT_DIR"

    # Path to the cloud knowledge-base task index used in lifelong learning.
    CLOUD_KB_INDEX = "CLOUD_KB_INDEX"

    # "True" / "False" string: whether the initial training round has completed.
    HAS_COMPLETED_INITIAL_TRAINING = "HAS_COMPLETED_INITIAL_TRAINING"

    # Floating-point threshold (as a string) for model-update trigger decisions.
    MODEL_THRESHOLD = "model_threshold"

    # Comparison operator string (">", "<", ">=", …) for model-update trigger decisions.
    OPERATOR = "operator"

    # Set to "TRUE" when ianvs runs a local (non-Kubernetes) benchmark.
    LOCAL_TEST = "LOCAL_TEST"
