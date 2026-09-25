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

"""Unit tests for parse_kwargs keyword-only parameter handling (Issue #597)."""

import unittest
from core.common.utils import parse_kwargs


class TestUtils(unittest.TestCase):
    """Test suite for Issue #597 utils.parse_kwargs defects."""

    def test_parse_kwargs_positional_and_kwonly(self):
        """Test parse_kwargs preserves both positional and keyword-only arguments."""
        def dummy_fn(a, b=1, *, metric_name="accuracy", threshold=0.8):
            pass

        kwargs = {
            "a": 10,
            "b": 20,
            "metric_name": "f1",
            "threshold": 0.9,
            "invalid_extra_param": "discard"
        }

        filtered = parse_kwargs(dummy_fn, **kwargs)
        expected = {"a": 10, "b": 20, "metric_name": "f1", "threshold": 0.9}
        self.assertEqual(filtered, expected)

    def test_parse_kwargs_with_varkw(self):
        """Test parse_kwargs returns all kwargs when function accepts **kwargs."""
        def varkw_fn(a, **kwargs):
            pass

        kwargs = {"a": 1, "extra1": "val1", "extra2": "val2"}
        filtered = parse_kwargs(varkw_fn, **kwargs)
        self.assertEqual(filtered, kwargs)

    def test_parse_kwargs_non_callable(self):
        """Test parse_kwargs returns kwargs unchanged if func is not callable."""
        kwargs = {"a": 1, "b": 2}
        filtered = parse_kwargs(None, **kwargs)
        self.assertEqual(filtered, kwargs)


if __name__ == "__main__":
    unittest.main()
