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

import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import types
import unittest

import numpy as np


class _Context:
    data_root = None

    @classmethod
    def get_parameters(cls, parameter, default=None):
        if parameter == "data_root":
            return cls.data_root
        return default


class _ClassFactory:
    @staticmethod
    def register(*args, **kwargs):
        del args, kwargs
        return lambda target: target


class _KittiDataloader:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.sample = {
            "dt": np.ones(5),
            "acc": np.zeros((5, 3)),
            "gyro": np.zeros((5, 3)),
            "velodyne": np.zeros((1, 4)),
            "init_pos": np.zeros((1, 3)),
            "init_rot": np.eye(3)[None, ...],
            "gt_pos": np.array(
                [[1.0, 0.0, 0.0],
                 [2.0, 0.0, 0.0],
                 [3.0, 0.0, 0.0],
                 [4.0, 0.0, 0.0],
                 [5.0, 0.0, 0.0]]
            ),
            "gt_rot": np.repeat(np.eye(3)[None, ...], 5, axis=0),
        }

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return self.sample

    def get_init_value(self):
        return {
            "pos": np.zeros((1, 3)),
            "rot": np.eye(3)[None, ...],
            "vel": np.zeros((1, 3)),
            "velodyne": [np.zeros((1, 4))],
        }


class _LLIOEstimator:
    def __init__(self, config):
        del config

    def process_batch(self, data):
        pose = np.eye(4)
        pose[:3, 3] = data["gt_pos"][-1]
        return pose


def _load_base_model():
    class_factory = types.ModuleType("sedna.common.class_factory")
    class_factory.ClassFactory = _ClassFactory
    class_factory.ClassType = types.SimpleNamespace(GENERAL="GENERAL")
    config = types.ModuleType("sedna.common.config")
    config.Context = _Context
    logger = types.ModuleType("core.common.log")
    logger.LOGGER = types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    estimator = types.ModuleType("llio_estimator")
    estimator.LLIOEstimator = _LLIOEstimator
    dataloader = types.ModuleType("kitti.dataloader")
    dataloader.KittiDataloader = _KittiDataloader
    dataloader.imu_collate = lambda data: data

    sys.modules.setdefault("sedna", types.ModuleType("sedna"))
    sys.modules.setdefault("sedna.common", types.ModuleType("sedna.common"))
    sys.modules["sedna.common.class_factory"] = class_factory
    sys.modules["sedna.common.config"] = config
    sys.modules["core.common.log"] = logger
    sys.modules["pykitti"] = types.ModuleType("pykitti")
    sys.modules["llio_estimator"] = estimator
    sys.modules["kitti"] = types.ModuleType("kitti")
    sys.modules["kitti.dataloader"] = dataloader

    module_path = (
        Path(__file__).parents[1]
        / "testalgorithms"
        / "llio_fusion"
        / "basemodel.py"
    )
    spec = importlib.util.spec_from_file_location("llio_basemodel_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BaseModel


class LLIOPredictTest(unittest.TestCase):
    def test_prediction_pairs_estimate_with_end_of_processed_window(self):
        base_model_class = _load_base_model()
        model = base_model_class.__new__(base_model_class)
        model.step_size = 5
        model.best_parameters = None
        model.gyro_std = 0.0032
        model.acc_std = 0.02
        model.voxel_size = 0.3
        model.icp_inlier_threshold = 0.5
        model.use_lidar_correction = True
        model.use_groundtruth_rot = False
        model.lidar_only_mode = False

        with TemporaryDirectory() as data_root:
            _Context.data_root = data_root
            Path(data_root, "test_index.txt").write_text(
                "2011_09_26/2011_09_26_drive_0001_sync\n",
                encoding="utf-8",
            )
            result = model.predict(None)

        self.assertEqual(result["estimated_poses"][0, 0, 3], 5.0)
        self.assertEqual(result["ground_truth_poses"][0, 0, 3], 5.0)
        self.assertEqual(result["sequence_lengths"], [1])

    def test_training_compares_estimate_with_end_of_processed_window(self):
        base_model_class = _load_base_model()
        model = base_model_class.__new__(base_model_class)
        model.step_size = 5
        model.gyro_std = 0.0032
        model.acc_std = 0.02
        model.voxel_size = 0.3
        model.icp_inlier_threshold = 0.5
        model.use_lidar_correction = True
        model.use_groundtruth_rot = False
        model.lidar_only_mode = False
        model.best_error = float("inf")
        model.best_parameters = None

        with TemporaryDirectory() as data_root:
            _Context.data_root = data_root
            Path(data_root, "train_index.txt").write_text(
                "2011_09_26/2011_09_26_drive_0001_sync\n",
                encoding="utf-8",
            )
            model.train(None)

        self.assertAlmostEqual(model.best_error, 0.0)


if __name__ == "__main__":
    unittest.main()
