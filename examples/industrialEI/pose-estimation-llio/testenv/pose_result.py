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
    homogeneous_row = np.array([0.0, 0.0, 0.0, 1.0])
    if (
        not np.allclose(ground_truth[:, 3, :], homogeneous_row)
        or not np.allclose(estimated[:, 3, :], homogeneous_row)
    ):
        raise ValueError("LLIO poses must have homogeneous bottom row [0, 0, 0, 1]")
    identity = np.eye(3)
    for poses in (ground_truth, estimated):
        rotations = poses[:, :3, :3]
        orthogonality = np.matmul(
            np.transpose(rotations, (0, 2, 1)), rotations
        )
        if (
            not np.allclose(orthogonality, identity, atol=1e-6)
            or not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-6)
        ):
            raise ValueError("LLIO poses must contain valid rotation matrices")

    return ground_truth, estimated


def iter_pose_sequences(result):
    """Yield aligned poses one sequence at a time."""
    ground_truth, estimated = unpack_pose_result(result)
    if "sequence_lengths" not in result:
        raise ValueError("LLIO inference result is missing: sequence_lengths")

    sequence_lengths = result["sequence_lengths"]
    if not isinstance(sequence_lengths, (list, tuple)) or not sequence_lengths:
        raise ValueError("LLIO sequence_lengths must be a non-empty list")
    if any(not isinstance(length, int) or length <= 0 for length in sequence_lengths):
        raise ValueError("LLIO sequence_lengths must contain positive integers")
    if sum(sequence_lengths) != len(ground_truth):
        raise ValueError(
            "LLIO sequence_lengths must account for every pose, "
            f"got {sum(sequence_lengths)} for {len(ground_truth)} poses"
        )

    start = 0
    for length in sequence_lengths:
        end = start + length
        yield ground_truth[start:end], estimated[start:end]
        start = end
