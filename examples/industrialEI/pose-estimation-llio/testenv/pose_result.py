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


def unpack_pose_result(result):
    """Return aligned ground-truth and estimated homogeneous poses."""
    if not isinstance(result, dict):
        raise ValueError("LLIO inference result must be a dictionary")

    required_keys = ("ground_truth_poses", "estimated_poses")
    missing_keys = [key for key in required_keys if key not in result]
    if missing_keys:
        raise ValueError(
            "LLIO inference result is missing: " + ", ".join(missing_keys)
        )

    ground_truth = np.asarray(result["ground_truth_poses"], dtype=float)
    estimated = np.asarray(result["estimated_poses"], dtype=float)

    if ground_truth.size == 0 and estimated.size == 0:
        raise ValueError("LLIO inference result contains no poses")
    if ground_truth.shape != estimated.shape:
        raise ValueError(
            "Ground-truth and estimated poses must have matching shapes, "
            f"got {ground_truth.shape} and {estimated.shape}"
        )
    if ground_truth.ndim != 3 or ground_truth.shape[1:] != (4, 4):
        raise ValueError(
            "LLIO poses must have shape (N, 4, 4), "
            f"got {ground_truth.shape}"
        )
    if not np.all(np.isfinite(ground_truth)) or not np.all(np.isfinite(estimated)):
        raise ValueError("LLIO poses must contain only finite values")

    return ground_truth, estimated
