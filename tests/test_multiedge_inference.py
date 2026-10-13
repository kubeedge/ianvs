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

"""Unit tests for MultiedgeInference paradigm."""

# pylint: disable=protected-access,unused-argument,missing-class-docstring,missing-function-docstring,wrong-import-position,too-few-public-methods

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.testcasecontroller.algorithm.paradigm.multiedge_inference.multiedge_inference import (
    MultiedgeInference,
)


class MockDataset:
    def __init__(self, x=None, need_other_info=False):
        self.x = x if x is not None else [1, 2, 3]
        self.test_url = "/tmp/test.txt"
        self.train_url = ""  # Pure inference benchmark has no train_url
        if need_other_info:
            self.need_other_info = True

    def load_data(self, url, data_type=None):
        if url == self.train_url and not url:
            raise ValueError("train_url is empty")
        return self


class MockJob:
    def __init__(self):
        self.loaded_model = None
        self.predict_input = None
        self.model_parallel = False

    def load(self, model, map_info=None):
        self.loaded_model = model

    def predict(self, x):
        self.predict_input = x
        return [0] * len(x) if isinstance(x, list) else [0]


class TestMultiedgeInference(unittest.TestCase):
    """Test suite for MultiedgeInference paradigm fixes."""

    def setUp(self):
        self.workspace = "/tmp/test_workspace"

    def test_inference_loads_only_test_url(self):
        """_inference must only load test_url and call job.predict(inference_dataset.x)."""
        paradigm = object.__new__(MultiedgeInference)
        paradigm.workspace = self.workspace
        paradigm.dataset = MockDataset(x=["img1.jpg", "img2.jpg"])

        mock_job = MockJob()
        res = paradigm._inference(mock_job, "model_checkpoint.pth")

        self.assertEqual(mock_job.loaded_model, "model_checkpoint.pth")
        self.assertEqual(mock_job.predict_input, ["img1.jpg", "img2.jpg"])
        self.assertEqual(res, [0, 0])

    def test_inference_with_need_other_info(self):
        """_inference should pass entire dataset if need_other_info attribute is present."""
        paradigm = object.__new__(MultiedgeInference)
        paradigm.workspace = self.workspace
        dataset = MockDataset(x=["img1.jpg"], need_other_info=True)
        paradigm.dataset = dataset

        mock_job = MockJob()
        paradigm._inference(mock_job, "model.pth")
        self.assertEqual(mock_job.predict_input, dataset)

    def test_inference_mp_with_need_other_info(self):
        """_inference_mp should support need_other_info dataset attribute."""
        paradigm = object.__new__(MultiedgeInference)
        paradigm.workspace = self.workspace
        dataset = MockDataset(x=["img1.jpg"], need_other_info=True)
        paradigm.dataset = dataset

        mock_job = MockJob()
        paradigm._inference_mp(mock_job, "/tmp/models", {"sub_1.onnx": "edge"})
        self.assertEqual(mock_job.predict_input, dataset)

    def test_partition_success_portable_paths(self):
        """_partition should create portable sub-model paths and valid map_info."""
        paradigm = object.__new__(MultiedgeInference)
        paradigm.workspace = self.workspace

        partition_point_list = [
            {"input_names": ["input_1"], "output_names": ["out_1"], "device_name": "edge"},
            {"input_names": ["out_1"], "output_names": ["output"], "device_name": "cloud"}
        ]

        sub_model_dir = os.path.join(self.workspace, "models")

        with patch("onnx.utils.extract_model") as mock_extract:
            models_dir, map_info = paradigm._partition(
                partition_point_list,
                "/tmp/base_model.onnx",
                sub_model_dir
            )

        self.assertEqual(models_dir, sub_model_dir)
        self.assertEqual(mock_extract.call_count, 2)
        self.assertEqual(map_info, {
            "sub_model_1.onnx": "edge",
            "sub_model_2.onnx": "cloud"
        })

    def test_partition_empty_list_raises_value_error(self):
        """_partition should raise ValueError when partition_point_list is empty or None."""
        paradigm = object.__new__(MultiedgeInference)
        paradigm.workspace = self.workspace

        with self.assertRaises(ValueError):
            paradigm._partition(None, "model.onnx", "/tmp/dir")

        with self.assertRaises(ValueError):
            paradigm._partition([], "model.onnx", "/tmp/dir")

    def test_partition_onnx_extraction_failure_raises_runtime_error(self):
        """_partition should raise RuntimeError when onnx extraction fails."""
        paradigm = object.__new__(MultiedgeInference)
        paradigm.workspace = self.workspace

        partition_point_list = [
            {"input_names": ["input_1"], "output_names": ["out_1"], "device_name": "edge"}
        ]

        with patch("onnx.utils.extract_model", side_effect=Exception("ONNX node mismatch")):
            with self.assertRaises(RuntimeError) as ctx:
                paradigm._partition(
                    partition_point_list,
                    "/tmp/broken_model.onnx",
                    os.path.join(self.workspace, "models")
                )
            self.assertIn("failed to extract ONNX sub-model", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
