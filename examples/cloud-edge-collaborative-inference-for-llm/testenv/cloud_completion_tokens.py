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

@ClassFactory.register(ClassType.GENERAL, alias="Cloud Completion Tokens")
def cloud_completion_tokens(_, y_pred):
    """Calculate the number of completion tokens generated by the cloud model.

    Parameters
    ----------
    _ :
        Ignored
    y_pred : list
        List of predictions from the JointInference paradigm

    Returns
    -------
    int
        Number of completion tokens generated by the cloud model
    """

    infer_res = [JointInferenceResult.from_list(*pred) for pred in y_pred]

    cloud_completion_tokens = sum([pred.cloud_result.completion_tokens for pred in infer_res])

    return cloud_completion_tokens