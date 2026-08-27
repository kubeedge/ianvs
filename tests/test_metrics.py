# Copyright 2026 The KubeEdge Authors.
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

"""Unit tests for system metrics defensive checks, calculation accuracy, and validation (Issue #647)."""

import unittest
import numpy as np

from core.testcasecontroller.metrics.metrics import (
    samples_transfer_ratio_func,
    task_avg_acc_func,
    forget_rate_func,
    bwt_func,
    fwt_func,
    matrix_func,
    get_metric_func,
)
from core.common.constant import SystemMetricType


class TestSystemMetrics(unittest.TestCase):
    """Test suite for Issue #647 system metrics defects."""

    def test_samples_transfer_ratio_defensive_and_math(self):
        """Test samples_transfer_ratio_func math accuracy and None handling."""
        # 1. Missing / None / Empty handling
        self.assertTrue(np.isnan(samples_transfer_ratio_func({})))
        self.assertTrue(np.isnan(samples_transfer_ratio_func(None)))
        self.assertTrue(np.isnan(samples_transfer_ratio_func({SystemMetricType.SAMPLES_TRANSFER_RATIO.value: None})))

        # 2. Zero inference samples (returns 0.0 without division by zero)
        self.assertEqual(samples_transfer_ratio_func({SystemMetricType.SAMPLES_TRANSFER_RATIO.value: [([], [])]}), 0.0)

        # 3. Accurate ratio calculation (e.g. 10 transfer samples out of 20 inference samples = 0.5, no +1 offset)
        info = [
            ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5]),
            ([11, 12, 13, 14, 15, 16, 17, 18, 19, 20], [6, 7, 8, 9, 10]),
        ]
        res = samples_transfer_ratio_func({SystemMetricType.SAMPLES_TRANSFER_RATIO.value: info})
        self.assertEqual(res, 0.5)

    def test_task_avg_acc_defensive(self):
        """Test task_avg_acc_func with valid, missing, and None dicts."""
        self.assertTrue(np.isnan(task_avg_acc_func({})))
        self.assertTrue(np.isnan(task_avg_acc_func({SystemMetricType.TASK_AVG_ACC.value: None})))
        self.assertTrue(np.isnan(task_avg_acc_func({SystemMetricType.TASK_AVG_ACC.value: {}})))

        res = task_avg_acc_func({SystemMetricType.TASK_AVG_ACC.value: {"accuracy": 0.8946}})
        self.assertEqual(res, 0.895)

    def test_forget_rate_defensive(self):
        """Test forget_rate_func with valid and empty inputs."""
        self.assertTrue(np.isnan(forget_rate_func({})))
        self.assertTrue(np.isnan(forget_rate_func({SystemMetricType.FORGET_RATE.value: None})))
        self.assertTrue(np.isnan(forget_rate_func({SystemMetricType.FORGET_RATE.value: []})))

        res = forget_rate_func({SystemMetricType.FORGET_RATE.value: [0.1, 0.2, 0.3]})
        self.assertEqual(res, 0.2)

    def test_bwt_fwt_matrix_defensive(self):
        """Test bwt, fwt, matrix functions handling missing Matrix info cleanly."""
        self.assertTrue(np.isnan(bwt_func({})))
        self.assertTrue(np.isnan(fwt_func({})))
        self.assertIsNone(matrix_func({}))

        self.assertTrue(np.isnan(bwt_func({SystemMetricType.MATRIX.value: None})))
        self.assertTrue(np.isnan(fwt_func({SystemMetricType.MATRIX.value: None})))
        self.assertIsNone(matrix_func({SystemMetricType.MATRIX.value: None}))

    def test_get_metric_func_validation(self):
        """Test get_metric_func returns built-in metric functions and raises descriptive ValueError for unknown metrics."""
        # Built-in metric resolution
        name, func = get_metric_func({"name": SystemMetricType.SAMPLES_TRANSFER_RATIO.value})
        self.assertEqual(name, SystemMetricType.SAMPLES_TRANSFER_RATIO.value)
        self.assertEqual(func, samples_transfer_ratio_func)

        # Invalid/Unsupported metric resolution without URL
        with self.assertRaises(ValueError) as ctx:
            get_metric_func({"name": "unsupported_custom_metric"})
        self.assertIn("Unsupported built-in metric 'unsupported_custom_metric'", str(ctx.exception))
        self.assertIn("Built-in supported metrics are", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
