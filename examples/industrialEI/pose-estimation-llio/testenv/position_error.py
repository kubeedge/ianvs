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

import numpy as np
from sedna.common.class_factory import ClassType, ClassFactory
from pose_result import unpack_pose_result

__all__ = ["position_error"]


@ClassFactory.register(ClassType.GENERAL, alias="position_error")
def position_error(y_true, y_pred, **kwargs):
    """
    Calculate position error metric for pose estimation.
    
    Args:
        y_true: Dataset labels supplied by Ianvs (unused by this example)
        y_pred: LLIO inference result containing paired ground-truth and
            estimated poses
        **kwargs: Additional arguments
        
    Returns:
        float: Average position error in meters
    """
    del y_true, kwargs
    ground_truth, estimated = unpack_pose_result(y_pred)
    translation_errors = np.linalg.norm(
        ground_truth[:, :3, 3] - estimated[:, :3, 3], axis=1
    )
    return float(np.mean(translation_errors))
