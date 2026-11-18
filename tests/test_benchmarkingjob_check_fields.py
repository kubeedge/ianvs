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

"""Unit tests for BenchmarkingJob._check_fields()"""

import pytest

from core.cmd.obj.benchmarkingjob import BenchmarkingJob


def test_empty_name_raises():
    with pytest.raises(ValueError, match="name"):
        BenchmarkingJob({"name": "", "test_object": {"type": "algorithms", "algorithms": [{}]}})


def test_wrong_name_type_raises():
    with pytest.raises(ValueError, match="name"):
        BenchmarkingJob({"name": 123, "test_object": {"type": "algorithms", "algorithms": [{}]}})


def test_empty_test_object_raises():
    with pytest.raises(ValueError, match="test_object"):
        BenchmarkingJob({"name": "benchmarkingjob", "test_object": {}})


def test_valid_config_does_not_raise():
    BenchmarkingJob(
        {
            "name": "benchmarkingjob",
            "test_object": {"type": "algorithms", "algorithms": [{}]},
        }
    )
