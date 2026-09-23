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

"""Unit tests for IncrementalLearning paradigm."""

# pylint: disable=protected-access,unused-argument,missing-class-docstring,missing-function-docstring,wrong-import-position,too-few-public-methods,unnecessary-lambda-assignment,unused-variable,line-too-long

import os
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.testcasecontroller.algorithm.paradigm.incremental_learning.incremental_learning import (
    IncrementalLearning,
)


class MockDataset:
    def __init__(self, x=None, y=None):
        self.x = x if x is not None else []
        self.y = y if y is not None else []
        self.train_url = "/tmp/train.txt"
        self.test_url = "/tmp/test.txt"

    def load_data(self, data_url, data_type=None):
        return self


class MockJob:
    def __init__(self, inference_return=None, evaluate_return=None):
        self.inference_return = inference_return
        self.evaluate_return = evaluate_return
        self.evaluate_metric_passed = None

    def inference(self, data):
        if self.inference_return is not None:
            return self.inference_return
        return [0], None, False

    def evaluate(self, eval_dataset, metric=None):
        self.evaluate_metric_passed = metric
        return self.evaluate_return or [{"metrics": {"accuracy": 0.9}}, {"metrics": {"accuracy": 0.8}}]


class TestIncrementalLearning(unittest.TestCase):
    """Test suite for IncrementalLearning paradigm bug fixes."""

    def setUp(self):
        self.workspace = "/tmp/test_workspace"

    def test_eval_passes_callable_metric(self):
        """_eval must unpack get_metric_func and pass a callable metric to job.evaluate."""
        paradigm = object.__new__(IncrementalLearning)
        paradigm.workspace = self.workspace
        paradigm.dataset = MockDataset()
        mock_job = MockJob()
        paradigm.build_paradigm_job = lambda _paradigm_type: mock_job
        paradigm.model_eval_config = {
            "model_metric": {
                "name": "accuracy",
                "url": None
            }
        }

        mock_metric_fn = lambda y_true, y_pred: 0.95
        with patch(
            "core.testcasecontroller.algorithm.paradigm.incremental_learning.incremental_learning.get_metric_func",
            return_value=("accuracy", mock_metric_fn)
        ):
            eval_results = paradigm._eval("new_model_url", "old_model_url", "/tmp/eval.txt")

        # Verify that the metric passed to job.evaluate is callable, not a tuple
        self.assertTrue(callable(mock_job.evaluate_metric_passed))
        self.assertNotIsInstance(mock_job.evaluate_metric_passed, tuple)
        self.assertEqual(mock_job.evaluate_metric_passed, mock_metric_fn)

    def test_get_train_dataset_success(self):
        """_get_train_dataset must write all hard example rows when all samples match uniquely."""
        paradigm = object.__new__(IncrementalLearning)
        paradigm.workspace = self.workspace
        paradigm.dataset_output_dir = lambda: os.path.dirname(os.path.abspath(__file__))

        hard_examples = [
            ("sample1.jpg", "/tmp/hard/new_sample1.jpg"),
            ("sample2.jpg", "/tmp/hard/new_sample2.jpg")
        ]

        data_labels = MockDataset(
            x=np.array(["sample1.jpg", "sample2.jpg"]),
            y=np.array(["cat", "dog"])
        )
        paradigm.dataset = MockDataset()
        paradigm.dataset.load_data = lambda _file, _type: data_labels

        train_dataset_file = paradigm._get_train_dataset(hard_examples, "/tmp/labels.txt")
        self.assertTrue(os.path.exists(train_dataset_file))

        with open(train_dataset_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]

        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0], "/tmp/hard/new_sample1.jpg cat")
        self.assertEqual(lines[1], "/tmp/hard/new_sample2.jpg dog")

        if os.path.exists(train_dataset_file):
            os.remove(train_dataset_file)

    def test_get_train_dataset_missing_or_ambiguous_raises(self):
        """_get_train_dataset must raise ValueError when samples are missing or ambiguous in data labels."""
        paradigm = object.__new__(IncrementalLearning)
        paradigm.workspace = self.workspace
        paradigm.dataset_output_dir = lambda: os.path.dirname(os.path.abspath(__file__))

        # Case 1: Missing sample
        hard_examples_missing = [
            ("valid.jpg", "/tmp/hard/new_valid.jpg"),
            ("missing.jpg", "/tmp/hard/new_missing.jpg")
        ]
        data_labels = MockDataset(
            x=np.array(["valid.jpg"]),
            y=np.array(["cat"])
        )
        paradigm.dataset = MockDataset()
        paradigm.dataset.load_data = lambda _file, _type: data_labels

        with self.assertRaises(ValueError) as ctx:
            paradigm._get_train_dataset(hard_examples_missing, "/tmp/labels.txt")
        self.assertIn("missing.jpg", str(ctx.exception))

        # Case 2: Ambiguous/duplicate sample in data_labels
        hard_examples_ambiguous = [
            ("duplicate.jpg", "/tmp/hard/new_duplicate.jpg")
        ]
        data_labels_dup = MockDataset(
            x=np.array(["duplicate.jpg", "duplicate.jpg"]),
            y=np.array(["cat", "dog"])
        )
        paradigm.dataset.load_data = lambda _file, _type: data_labels_dup

        with self.assertRaises(ValueError) as ctx:
            paradigm._get_train_dataset(hard_examples_ambiguous, "/tmp/labels.txt")
        self.assertIn("duplicate.jpg", str(ctx.exception))

    def test_inference_predictions(self):
        """_inference must collect dict and non-dict predictions safely."""
        paradigm = object.__new__(IncrementalLearning)
        paradigm.workspace = self.workspace
        paradigm._prepare_inference = lambda _model, _rounds: "/tmp/hard_examples"
        paradigm.dataset = MockDataset(x=["sample1.jpg", "sample2.jpg"])

        # Case 1: Estimator returns list prediction: ([1], None, False)
        mock_job = MockJob(inference_return=([1], None, False))
        paradigm.build_paradigm_job = lambda _paradigm_type: mock_job

        inference_results, hard_examples = paradigm._inference("model_url", "/tmp/test.txt", 1)
        self.assertEqual(inference_results, {"sample1.jpg": [1], "sample2.jpg": [1]})
        self.assertEqual(hard_examples, [])

        # Case 2: Estimator returns dict prediction: ({"sample1.jpg": [1]}, None, False)
        mock_job_dict = MockJob(inference_return=({"sample1.jpg": [1]}, None, False))
        paradigm.build_paradigm_job = lambda _paradigm_type: mock_job_dict

        inference_results_dict, _ = paradigm._inference("model_url", "/tmp/test.txt", 1)
        self.assertEqual(inference_results_dict, {"sample1.jpg": [1]})

    def test_trigger_model_update_valid_cases(self):
        """_trigger_model_update should correctly evaluate delta with comparison operator."""
        paradigm = object.__new__(IncrementalLearning)
        paradigm.model_eval_config = {
            "model_metric": {"name": "accuracy"},
            "operator": ">",
            "threshold": 0.05
        }

        # Case 1: Standard dict with 'metrics'
        eval_results_1 = [
            {"metrics": {"accuracy": 0.95}},
            {"metrics": {"accuracy": 0.85}}
        ]
        # delta = 0.95 - 0.85 = 0.10 > 0.05 -> True
        self.assertTrue(paradigm._trigger_model_update(eval_results_1))

        # Case 2: Direct float values
        eval_results_2 = [0.80, 0.82]
        # delta = -0.02 > 0.05 -> False
        self.assertFalse(paradigm._trigger_model_update(eval_results_2))

    def test_trigger_model_update_missing_metrics_raises(self):
        """_trigger_model_update should raise RuntimeError when metric cannot be extracted."""
        paradigm = object.__new__(IncrementalLearning)
        paradigm.model_eval_config = {
            "model_metric": {"name": "accuracy"},
            "operator": ">",
            "threshold": 0.05
        }

        # Case 1: Missing key inside nested metrics dict
        eval_results_nested_missing = [
            {"metrics": {"f1_score": 0.95}},
            {"metrics": {"accuracy": 0.85}}
        ]
        with self.assertRaises(RuntimeError):
            paradigm._trigger_model_update(eval_results_nested_missing)

        # Case 2: Empty result dictionaries
        eval_results_empty = [{}, {}]
        with self.assertRaises(RuntimeError):
            paradigm._trigger_model_update(eval_results_empty)

        # Case 3: Direct dictionaries missing the configured metric
        eval_results_direct_missing = [
            {"f1_score": 0.95},
            {"loss": 0.15}
        ]
        with self.assertRaises(RuntimeError):
            paradigm._trigger_model_update(eval_results_direct_missing)

        # Case 4: Nonnumeric metric values (strings, None)
        eval_results_nonnumeric_nested = [
            {"metrics": {"accuracy": "high"}},
            {"metrics": {"accuracy": 0.85}}
        ]
        with self.assertRaises(RuntimeError):
            paradigm._trigger_model_update(eval_results_nonnumeric_nested)

        eval_results_nonnumeric_direct = [
            {"accuracy": None},
            {"accuracy": 0.85}
        ]
        with self.assertRaises(RuntimeError):
            paradigm._trigger_model_update(eval_results_nonnumeric_direct)


if __name__ == "__main__":
    unittest.main()
