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

"""Unit tests for Rank._check_fields()"""

import pytest

from core.storymanager.rank.rank import Rank


def test_sort_by_empty_or_omitted_allowed():
    Rank(
        {
            "sort_by": [],
            "visualization": {"mode": "selected_only", "method": "print_table"},
            "selected_dataitem": {
                "paradigms": ["all"],
                "modules": ["all"],
                "hyperparameters": ["all"],
                "metrics": ["all"],
            },
        }
    )


def test_sort_by_wrong_type_raises():
    with pytest.raises(ValueError, match="sort_by"):
        Rank({"sort_by": "not-a-list"})


def test_visualization_empty_raises():
    with pytest.raises(ValueError, match="visualization"):
        Rank({"sort_by": [{"accuracy": "descend"}], "visualization": {}})


def test_selected_dataitem_empty_raises():
    with pytest.raises(ValueError, match="selected_dataitem"):
        Rank(
            {
                "sort_by": [{"accuracy": "descend"}],
                "visualization": {"mode": "selected_only", "method": "print_table"},
                "selected_dataitem": {},
            }
        )


def test_save_mode_wrong_type_raises():
    with pytest.raises(ValueError, match="save_mode"):
        Rank(
            {
                "sort_by": [{"accuracy": "descend"}],
                "visualization": {"mode": "selected_only", "method": "print_table"},
                "selected_dataitem": {
                    "paradigms": ["all"],
                    "modules": ["all"],
                    "hyperparameters": ["all"],
                    "metrics": ["all"],
                },
                "save_mode": ["not", "a", "str"],
            }
        )


def test_save_mode_empty_raises():
    with pytest.raises(ValueError, match="save_mode"):
        Rank(
            {
                "sort_by": [{"accuracy": "descend"}],
                "visualization": {"mode": "selected_only", "method": "print_table"},
                "selected_dataitem": {
                    "paradigms": ["all"],
                    "modules": ["all"],
                    "hyperparameters": ["all"],
                    "metrics": ["all"],
                },
                "save_mode": "",
            }
        )


def test_valid_config_does_not_raise():
    Rank(
        {
            "sort_by": [{"accuracy": "descend"}],
            "visualization": {"mode": "selected_only", "method": "print_table"},
            "selected_dataitem": {
                "paradigms": ["all"],
                "modules": ["all"],
                "hyperparameters": ["all"],
                "metrics": ["all"],
            },
            "save_mode": "selected_and_all",
        }
    )
