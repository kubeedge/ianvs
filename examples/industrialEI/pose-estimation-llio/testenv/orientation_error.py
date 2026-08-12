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

__all__ = ["orientation_error"]


@ClassFactory.register(ClassType.GENERAL, alias="orientation_error")
def orientation_error(y_true, y_pred, **kwargs):
    """
    Calculate orientation error metric for pose estimation.
    
    Args:
        y_true: Dataset labels supplied by Ianvs (unused by this example)
        y_pred: LLIO inference result containing paired ground-truth and
            estimated poses
        **kwargs: Additional arguments
        
    Returns:
        float: Average orientation error in degrees
    """
    del y_true, kwargs
    ground_truth, estimated = unpack_pose_result(y_pred)
    relative_rotations = np.matmul(
        np.transpose(ground_truth[:, :3, :3], (0, 2, 1)),
        estimated[:, :3, :3],
    )
    traces = np.trace(relative_rotations, axis1=1, axis2=2)
    angles = np.arccos(np.clip((traces - 1.0) / 2.0, -1.0, 1.0))
    return float(np.mean(np.degrees(angles)))
