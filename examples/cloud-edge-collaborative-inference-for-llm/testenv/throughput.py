# Copyright 2024 The KubeEdge Authors.
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

from sedna.common.class_factory import ClassType, ClassFactory
from result_parser import JointInferenceResult

@ClassFactory.register(ClassType.GENERAL, alias="Throughput")
def throughput(_, y_pred):
    """Calculate the Throughput of the system.

    Parameters
    ----------
    _ :
        Ignored
    y_pred : list
        List of predictions from the JointInference paradigm

    Returns
    -------
    float
        Average Throughput (token/s) of the system
    """

    infer_res = [JointInferenceResult.from_list(*pred) for pred in y_pred]

    average_itl = sum([pred.result.internal_token_latency for pred in infer_res]) / len(infer_res)

    average_throughput = 1 / average_itl

    return round(average_throughput,2)