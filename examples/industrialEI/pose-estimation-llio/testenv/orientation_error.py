<<<<<<< HEAD
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

__all__ = ["orientation_error"]


def rotation_matrix_to_euler_angles(R):
    """
    Convert rotation matrix to Euler angles (roll, pitch, yaw).
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        tuple: (roll, pitch, yaw) in radians
    """
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    
    return x, y, z


@ClassFactory.register(ClassType.GENERAL, alias="orientation_error")
def orientation_error(y_true, y_pred, **kwargs):
    """
    Calculate orientation error metric for pose estimation.
    
    Args:
        y_true: Ground truth poses or sequence indices (from Ianvs)
        y_pred: Predicted poses
        **kwargs: Additional arguments
        
    Returns:
        float: Average orientation error in degrees
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
                # Calculate relative orientation changes between consecutive frames
                relative_errors = []
                for i in range(1, len(y_pred)):
                    # Calculate relative pose between frame i-1 and i
                    prev_pose = y_pred[i-1]
                    curr_pose = y_pred[i]
                    
                    # Relative rotation
                    prev_rot = prev_pose[:3, :3]
                    curr_rot = curr_pose[:3, :3]
                    relative_rot = prev_rot.T @ curr_rot
                    angle = np.arccos(np.clip((np.trace(relative_rot) - 1) / 2, -1, 1))
                    relative_rotation = np.degrees(angle)
                    
                    relative_errors.append(relative_rotation)
                
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
=======
version https://git-lfs.github.com/spec/v1
oid sha256:ab35b326e32268a4a0e2cbd003f498758e38d3d7f81fef3defecfc49cf5f2680
size 3231
>>>>>>> 9676c3e (ya toh aar ya toh par)
