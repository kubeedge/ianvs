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

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.testcasecontroller.metrics.metrics import (
    samples_transfer_ratio_func,
    task_avg_acc_func,
    forget_rate_func,
    bwt_func,
    fwt_func,
    matrix_func,
    compute,
    get_metric_func,
)


def test_samples_transfer_ratio_func():
    # Empty / None info returns nan
    assert np.isnan(samples_transfer_ratio_func(None))
    assert np.isnan(samples_transfer_ratio_func({}))

    # Valid data calculation (exact ratio without artificial +1 denominator offset)
    info = {
        "samples_transfer_ratio": [
            ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5]),  # 10 inference, 5 transfer
        ]
    }
    assert samples_transfer_ratio_func(info) == 0.5


def test_task_avg_acc_func():
    assert np.isnan(task_avg_acc_func(None))
    assert np.isnan(task_avg_acc_func({}))
    assert task_avg_acc_func({"task_avg_acc": {"accuracy": 0.8567}}) == 0.857


def test_forget_rate_func():
    assert np.isnan(forget_rate_func(None))
    assert np.isnan(forget_rate_func({}))
    assert np.isnan(forget_rate_func({"forget_rate": []}))
    assert forget_rate_func({"forget_rate": [0.1, 0.2, 0.3]}) == 0.2


def test_compute_and_bwt_fwt_matrix_func():
    # Invalid or empty matrix returns 3-tuple with [] and NaNs
    res, bwt, fwt = compute("test", None)
    assert res == []
    assert np.isnan(bwt)
    assert np.isnan(fwt)

    # Single-row matrix [[]] (length <= 1) should not trigger ZeroDivisionError
    res_single, bwt_single, fwt_single = compute("test", [[]])
    assert res_single == []
    assert np.isnan(bwt_single)
    assert np.isnan(fwt_single)

    res_empty, bwt_empty, fwt_empty = compute("test", [])
    assert res_empty == []
    assert np.isnan(bwt_empty)
    assert np.isnan(fwt_empty)

    # Missing info in bwt/fwt/matrix functions returns nan or empty dict
    assert np.isnan(bwt_func(None))
    assert np.isnan(fwt_func(None))
    assert matrix_func(None) == {}


def test_get_metric_func_unsupported_name():
    with pytest.raises(ValueError, match="not found in built-in metrics"):
        get_metric_func({"name": "unsupported_custom_metric_xyz"})
