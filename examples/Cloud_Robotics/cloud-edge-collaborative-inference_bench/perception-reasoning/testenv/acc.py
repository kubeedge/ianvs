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

import numpy as np
from sedna.common.class_factory import ClassType, ClassFactory


from sedna.common.class_factory import ClassType, ClassFactory

__all__ = ["pa"]

@ClassFactory.register(ClassType.GENERAL, alias="PA")
def pa(y_true, y_pred):
    """Calculate the pixel accuracy (PA) for semantic segmentation.

    Parameters
    ----------
    y_true : list
        Ground truth masks (as flat lists or arrays)
    y_pred : list
        Predicted masks (as flat lists or arrays)

    Returns
    -------
    float
        The pixel accuracy (%)
    """
    # Flatten masks and compare pixel-wise
    total = 0
    correct = 0
    for true_mask, pred_mask in zip(y_true, y_pred):
        true_mask = list(true_mask)
        pred_mask = list(pred_mask)
        min_len = min(len(true_mask), len(pred_mask))
        total += min_len
        correct += sum([1 for i in range(min_len) if true_mask[i] == pred_mask[i]])
    if total == 0:
        return 0.0
    return round(100 * correct / total, 2)
