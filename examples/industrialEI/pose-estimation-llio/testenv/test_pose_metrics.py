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

import sys
import types
import unittest

import numpy as np


if "sedna.common.class_factory" not in sys.modules:
    class _ClassFactory:
        @staticmethod
        def register(*args, **kwargs):
            del args, kwargs
            return lambda func: func

    class_factory = types.ModuleType("sedna.common.class_factory")
    class_factory.ClassFactory = _ClassFactory
    class_factory.ClassType = types.SimpleNamespace(GENERAL="GENERAL")
    sys.modules.setdefault("sedna", types.ModuleType("sedna"))
    sys.modules.setdefault("sedna.common", types.ModuleType("sedna.common"))
    sys.modules["sedna.common.class_factory"] = class_factory

from orientation_error import orientation_error
from position_error import position_error
from trajectory_consistency import trajectory_consistency


def _pose_result(ground_truth, estimated):
    return {
        "ground_truth_poses": np.asarray(ground_truth),
        "estimated_poses": np.asarray(estimated),
    }


def _translated(x=0.0, y=0.0, z=0.0):
    pose = np.eye(4)
    pose[:3, 3] = [x, y, z]
    return pose


def _rotated_z(degrees):
    radians = np.radians(degrees)
    cosine, sine = np.cos(radians), np.sin(radians)
    pose = np.eye(4)
    pose[:3, :3] = [
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ]
    return pose


class PoseMetricTest(unittest.TestCase):
    def test_identical_poses_have_zero_error(self):
        poses = [np.eye(4), _translated(x=1.0), _rotated_z(90.0)]
        result = _pose_result(poses, poses)

        self.assertAlmostEqual(position_error(None, result), 0.0)
        self.assertAlmostEqual(orientation_error(None, result), 0.0)

    def test_position_error_compares_matching_poses(self):
        result = _pose_result(
            [np.eye(4), _translated(x=1.0)],
            [_translated(x=1.0), _translated(x=2.0)],
        )

        self.assertAlmostEqual(position_error(None, result), 1.0)

    def test_orientation_error_compares_matching_poses(self):
        result = _pose_result(
            [np.eye(4), _rotated_z(90.0)],
            [_rotated_z(90.0), _rotated_z(180.0)],
        )

        self.assertAlmostEqual(orientation_error(None, result), 90.0)

    def test_trajectory_consistency_uses_paired_poses(self):
        ground_truth = [np.eye(4), _translated(x=1.0), _translated(x=2.0)]
        result = _pose_result(ground_truth, ground_truth)

        self.assertAlmostEqual(trajectory_consistency(None, result), 1.0)

    def test_mismatched_lengths_are_rejected(self):
        result = _pose_result([np.eye(4)], [np.eye(4), np.eye(4)])

        with self.assertRaisesRegex(ValueError, "matching shapes"):
            position_error(None, result)

    def test_malformed_pose_is_rejected(self):
        result = _pose_result([np.eye(3)], [np.eye(3)])

        with self.assertRaisesRegex(ValueError, "shape \\(N, 4, 4\\)"):
            orientation_error(None, result)

    def test_empty_result_is_rejected(self):
        result = _pose_result([], [])

        with self.assertRaisesRegex(ValueError, "contains no poses"):
            position_error(None, result)


if __name__ == "__main__":
    unittest.main()
