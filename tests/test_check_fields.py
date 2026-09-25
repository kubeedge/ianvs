# Copyright 2024 The KubeEdge Authors.
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

import pytest
from core.storymanager.rank import Rank
from core.cmd.obj.benchmarkingjob import BenchmarkingJob
from core.testcasecontroller.algorithm.algorithm import Algorithm
from core.testcasecontroller.algorithm.module import Module


def test_rank_check_fields_invalid_types():
    valid_sort_by = [{"acc": "ascend"}]
    valid_vis = {"mode": "selected_only", "method": "print_table"}

    # Test sort_by invalid type (string instead of list)
    with pytest.raises(ValueError, match="sort_by"):
        Rank({"sort_by": "accuracy"})

    # Test visualization invalid type (list instead of dict)
    with pytest.raises(ValueError, match="visualization"):
        Rank({"sort_by": valid_sort_by, "visualization": ["invalid"]})

    # Test selected_dataitem invalid type (list instead of dict)
    with pytest.raises(ValueError, match="selected_dataitem"):
        Rank({"sort_by": valid_sort_by, "visualization": valid_vis, "selected_dataitem": "invalid"})

    # Test save_mode invalid type (list instead of string)
    with pytest.raises(ValueError, match="save_mode"):
        Rank({
            "sort_by": valid_sort_by,
            "visualization": valid_vis,
            "selected_dataitem": {
                "paradigms": ["all"],
                "modules": ["all"],
                "metrics": ["all"]
            },
            "save_mode": ["invalid"]
        })


def test_benchmarkingjob_check_fields_invalid_types():
    # Test name invalid type (int instead of string)
    with pytest.raises(ValueError, match="name"):
        BenchmarkingJob({"name": 123})

    # Test test_object invalid type (string instead of dict)
    with pytest.raises(ValueError, match="test_object"):
        BenchmarkingJob({"name": "test_job", "test_object": "invalid"})


def test_algorithm_check_fields_invalid_types():
    # Test name invalid type (int instead of string)
    with pytest.raises(ValueError, match="name"):
        Algorithm(123, {"algorithm": {"name": 123, "paradigm_type": "singletasklearning"}})

    # Test paradigm_type invalid type (int instead of string)
    with pytest.raises(ValueError, match="paradigm"):
        Algorithm("test_alg", {"algorithm": {"name": "test_alg", "paradigm_type": 123}})


def test_module_check_fields_invalid_types():
    # Test module type invalid type (int instead of string)
    with pytest.raises(ValueError, match="module type"):
        Module({"type": 123, "name": "test_mod"})

    # Test module name invalid type (int instead of string)
    with pytest.raises(ValueError, match="module name"):
        Module({"type": "basemodel", "name": 123})
