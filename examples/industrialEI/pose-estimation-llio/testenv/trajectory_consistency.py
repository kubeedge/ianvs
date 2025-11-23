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

__all__ = ["trajectory_consistency"]


@ClassFactory.register(ClassType.GENERAL, alias="trajectory_consistency")
def trajectory_consistency(y_true, y_pred, **kwargs):
    """
    Calculate trajectory consistency metric for pose estimation.
    Evaluates trajectory smoothness and drift characteristics.
    
    Args:
        y_true: Ground truth poses in format [N, 4, 4] (homogeneous transformation matrices)
        y_pred: Predicted poses in format [N, 4, 4] (homogeneous transformation matrices)  
        **kwargs: Additional arguments
        
    Returns:
        float: Trajectory consistency score (higher is better, range 0-1)
    """
    # Handle length mismatch by truncating to shorter length
    min_length = min(len(y_true), len(y_pred))
    if min_length < 3:
        # Need at least 3 poses to evaluate trajectory consistency
        return 1.0
    
    y_true = y_true[:min_length]
    y_pred = y_pred[:min_length]
    
    # Convert to numpy arrays if they aren't already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Extract position components
    if y_true.ndim == 3 and y_true.shape[-2:] == (4, 4):
        # Homogeneous transformation matrices - extract translation
        true_positions = y_true[:, :3, 3]
        pred_positions = y_pred[:, :3, 3]
    elif y_true.ndim == 2 and y_true.shape[-1] >= 3:
        # Direct position format - take first 3 columns as position
        true_positions = y_true[:, :3]
        pred_positions = y_pred[:, :3]
    else:
        return 1.0
    
    # Calculate velocity vectors (differences between consecutive positions)
    true_velocities = np.diff(true_positions, axis=0)
    pred_velocities = np.diff(pred_positions, axis=0)
    
    # Calculate acceleration vectors (differences between consecutive velocities)
    true_accelerations = np.diff(true_velocities, axis=0)
    pred_accelerations = np.diff(pred_velocities, axis=0)
    
    # Velocity consistency: Compare velocity magnitudes and directions
    true_vel_magnitudes = np.linalg.norm(true_velocities, axis=1)
    pred_vel_magnitudes = np.linalg.norm(pred_velocities, axis=1)
    
    # Avoid division by zero
    vel_consistency = 1.0
    if len(true_vel_magnitudes) > 0:
        vel_magnitude_diff = np.abs(true_vel_magnitudes - pred_vel_magnitudes)
        max_vel_magnitude = np.maximum(true_vel_magnitudes, pred_vel_magnitudes)
        max_vel_magnitude = np.where(max_vel_magnitude == 0, 1e-6, max_vel_magnitude)
        vel_consistency = 1.0 - np.mean(vel_magnitude_diff / max_vel_magnitude)
    
    # Acceleration consistency: Smoothness evaluation
    acc_consistency = 1.0
    if len(true_accelerations) > 0:
        true_acc_magnitudes = np.linalg.norm(true_accelerations, axis=1)
        pred_acc_magnitudes = np.linalg.norm(pred_accelerations, axis=1)
        
        acc_magnitude_diff = np.abs(true_acc_magnitudes - pred_acc_magnitudes)
        max_acc_magnitude = np.maximum(true_acc_magnitudes, pred_acc_magnitudes)
        max_acc_magnitude = np.where(max_acc_magnitude == 0, 1e-6, max_acc_magnitude)
        acc_consistency = 1.0 - np.mean(acc_magnitude_diff / max_acc_magnitude)
    
    # Path deviation: Calculate area between trajectories
    path_consistency = 1.0
    if len(true_positions) > 2:
        # Calculate cumulative distances along both trajectories
        true_distances = np.cumsum(np.linalg.norm(np.diff(true_positions, axis=0), axis=1))
        pred_distances = np.cumsum(np.linalg.norm(np.diff(pred_positions, axis=0), axis=1))
        
        # Compare total path lengths
        true_total_dist = true_distances[-1] if len(true_distances) > 0 else 0
        pred_total_dist = pred_distances[-1] if len(pred_distances) > 0 else 0
        
        if true_total_dist > 0:
            path_length_ratio = min(pred_total_dist / true_total_dist, true_total_dist / pred_total_dist)
            path_consistency = path_length_ratio
    
    # Drift evaluation: Compare start and end point alignment
    drift_consistency = 1.0
    if len(true_positions) > 1:
        true_displacement = true_positions[-1] - true_positions[0]
        pred_displacement = pred_positions[-1] - pred_positions[0]
        
        displacement_error = np.linalg.norm(true_displacement - pred_displacement)
        true_displacement_magnitude = np.linalg.norm(true_displacement)
        
        if true_displacement_magnitude > 0:
            drift_consistency = 1.0 - min(displacement_error / true_displacement_magnitude, 1.0)
    
    # Combine all consistency measures
    # Weight: velocity (0.3), acceleration (0.2), path (0.3), drift (0.2)
    total_consistency = (
        0.3 * vel_consistency +
        0.2 * acc_consistency +
        0.3 * path_consistency +
        0.2 * drift_consistency
    )
    
    # Clamp to [0, 1] range
    total_consistency = np.clip(total_consistency, 0.0, 1.0)
    
    return float(total_consistency) 
=======
version https://git-lfs.github.com/spec/v1
oid sha256:1d14c1800a313d195aac6796586c617b91a23a9a3395365452ea0c9663122360
size 5583
>>>>>>> 9676c3e (ya toh aar ya toh par)
