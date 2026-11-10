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

"""Tests for the BWT and FWT lifelong learning metrics."""

import math

import pytest

from core.testcasecontroller.metrics.metrics import compute


def _to_matrix(scores):
    return [[{"accuracy": value} for value in row] for row in scores]


def test_bwt_is_negative_when_earlier_tasks_are_forgotten():
    # Row 0: before learning; row i: after learning task i. Three tasks.
    scores = [
        [0.10, 0.10, 0.10],
        [0.90, 0.30, 0.20],
        [0.70, 0.90, 0.40],
        [0.50, 0.70, 0.90],
    ]
    _, bwt, fwt = compute("all", _to_matrix(scores))
    # BWT = ((0.5 - 0.9) + (0.7 - 0.9)) / 2
    assert bwt == pytest.approx(-0.3)
    # FWT = ((0.3 - 0.1) + (0.4 - 0.1)) / 2
    assert fwt == pytest.approx(0.25)


def test_bwt_is_zero_without_forgetting():
    scores = [
        [0.1, 0.1, 0.1],
        [0.9, 0.1, 0.1],
        [0.9, 0.9, 0.1],
        [0.9, 0.9, 0.9],
    ]
    _, bwt, fwt = compute("all", _to_matrix(scores))
    assert bwt == pytest.approx(0.0)
    assert fwt == pytest.approx(0.0)


def test_single_task_has_no_transfer():
    _, bwt, fwt = compute("all", _to_matrix([[0.1], [0.8]]))
    assert math.isnan(bwt)
    assert math.isnan(fwt)
