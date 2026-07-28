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

from sedna.common.class_factory import ClassType, ClassFactory

__all__ = ["acc"]

def get_last_letter(input_string):
    if not input_string or not any(char.isalpha() for char in input_string):
        return None
    
    for char in reversed(input_string):
        if 'A' <= char <= 'D':
            return char

    return None


@ClassFactory.register(ClassType.GENERAL, alias="acc")
def acc(y_true, y_pred):
    y_pred = [get_last_letter(pred) for pred in y_pred]
    y_true = [get_last_letter(pred) for pred in y_true]
        
    same_elements = [y_pred[i] == y_true[i] for i in range(len(y_pred))]

    acc = sum(same_elements) / len(same_elements)
    
    return acc
