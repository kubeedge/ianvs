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

"""Tests for rank configuration validation."""

import pytest

from core.storymanager.rank.rank import Rank


VALID_RANK_CONFIG = {
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


def test_rank_accepts_valid_config():
    """Rank should accept the documented string save_mode format."""
    rank = Rank(VALID_RANK_CONFIG)

    assert rank.save_mode == "selected_and_all"


@pytest.mark.parametrize(
    "field,value,error_match",
    [
        ("sort_by", {}, "sort_by"),
        ("visualization", [], "visualization"),
        ("selected_dataitem", [], "selected_dataitem"),
        ("save_mode", [], "save_mode"),
    ],
)
def test_rank_rejects_invalid_field_types(field, value, error_match):
    """Rank should reject non-empty values with the wrong config type."""
    config = VALID_RANK_CONFIG.copy()
    config[field] = value

    with pytest.raises(ValueError, match=error_match):
        Rank(config)
