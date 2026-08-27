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

"""Unit tests for Module._check_fields()"""

import pytest

from core.testcasecontroller.algorithm.module.module import Module


def test_empty_type_raises():
    with pytest.raises(ValueError, match="type"):
        Module({"type": "", "name": "basemodel"})


def test_wrong_type_type_raises():
    with pytest.raises(ValueError, match="type"):
        Module({"type": 123, "name": "basemodel"})


def test_empty_name_raises():
    with pytest.raises(ValueError, match="name"):
        Module({"type": "basemodel", "name": ""})


def test_valid_config_does_not_raise():
    Module({"type": "basemodel", "name": "basemodel"})
