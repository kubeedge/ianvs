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

__all__ = ["acc"]

def get_last_letter(input_string):
    """Extract the prediction from the completion. This function is used when caching the responses.
    """
    if not input_string or not any(char.isalpha() for char in input_string):
        return None
    # Find the last letter in the string
    for char in reversed(input_string):
        if 'A' <= char <= 'D':
            return char
    return None

@ClassFactory.register(ClassType.GENERAL, alias="Accuracy")
def acc(y_true, y_pred):
    """Calculate the accuracy.

    Parameters
    ----------
    y_true : list
        Ground truth
    y_pred : list
        List of predictions from the JointInference paradigm

    Returns
    -------
    float
        The accuracy (%)
    """

    infer_res = [JointInferenceResult.from_list(*pred) for pred in y_pred]

    if not infer_res or not y_true:
        return 0.0

    y_pred = [get_last_letter(pred.result.completion) for pred in infer_res]
    y_true = [get_last_letter(y) for y in y_true]

    same_elements = [y_pred[i] == y_true[i] for i in range(min(len(y_pred), len(y_true)))]

    if not same_elements:
        return 0.0

    acc = sum(same_elements) / len(same_elements)

    return round(acc * 100, 2)
