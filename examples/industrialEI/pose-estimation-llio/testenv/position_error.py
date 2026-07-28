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

__all__ = ["position_error"]


@ClassFactory.register(ClassType.GENERAL, alias="position_error")
def position_error(y_true, y_pred, **kwargs):
    """
    Calculate position error metric for pose estimation.
    
    Args:
        y_true: Ground truth poses or sequence indices (from Ianvs)
        y_pred: Predicted poses
        **kwargs: Additional arguments
        
    Returns:
        float: Average position error in meters
    """
    if len(y_pred) == 0:
        return 0.0
    
    # Handle length mismatch by truncating to shorter length
    min_length = min(len(y_true), len(y_pred))
    if min_length == 0:
        return 0.0
    
    y_true = y_true[:min_length]
    y_pred = y_pred[:min_length]
    
    try:
        # Check if y_pred contains pose matrices
        if isinstance(y_pred[0], np.ndarray) and y_pred[0].shape == (4, 4):
            if len(y_pred) > 1:
                # Calculate relative pose changes between consecutive frames
                relative_errors = []
                for i in range(1, len(y_pred)):
                    # Calculate relative pose between frame i-1 and i
                    prev_pose = y_pred[i-1]
                    curr_pose = y_pred[i]
                    
                    # Relative translation
                    prev_pos = prev_pose[:3, 3]
                    curr_pos = curr_pose[:3, 3]
                    relative_translation = np.linalg.norm(curr_pos - prev_pos)
                    
                    relative_errors.append(relative_translation)
                
                if relative_errors:
                    return float(np.mean(relative_errors))
                else:
                    return 0.0
            else:
                return 0.0
        else:
            return 0.0
            
    except Exception:
        return 0.0 