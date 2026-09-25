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

"""Unit tests for utility functions."""

# pylint: disable=unused-argument,missing-function-docstring,wrong-import-position

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.common.utils import parse_kwargs


def test_parse_kwargs_positional_and_kwonlyargs():
    def sample_func(a, b=1, *, c=2, d=3):
        return a + b + c + d

    kwargs = {"a": 10, "b": 20, "c": 30, "d": 40, "extra": 50}
    filtered = parse_kwargs(sample_func, **kwargs)

    assert filtered == {"a": 10, "b": 20, "c": 30, "d": 40}
    assert "extra" not in filtered


def test_parse_kwargs_varkw():
    def sample_func_with_kwargs(a, **kwargs):
        pass

    def sample_func_with_opts(a, **opts):
        pass

    kwargs = {"a": 1, "b": 2, "c": 3}
    assert parse_kwargs(sample_func_with_kwargs, **kwargs) == kwargs
    assert parse_kwargs(sample_func_with_opts, **kwargs) == kwargs


def test_parse_kwargs_non_callable():
    assert parse_kwargs("not_a_func", x=1, y=2) == {"x": 1, "y": 2}


def test_parse_kwargs_positional_only():
    def posonly_func(x, /, y, *, z=3):
        return x + y + z

    kwargs = {"x": 1, "y": 2, "z": 3, "extra": 4}
    filtered = parse_kwargs(posonly_func, **kwargs)
    assert filtered == {"y": 2, "z": 3}


def test_parse_kwargs_unintrospectable_and_builtins():
    # Built-in types with non-standard signatures fall back safely to returning kwargs
    assert parse_kwargs(dict, a=1, b=2) == {"a": 1, "b": 2}
    # Built-in functions with only positional-only parameters fall back to returning kwargs
    assert parse_kwargs(len, a=1) == {"a": 1}
    # Built-in functions with keyword-only parameters filter valid keywords
    assert parse_kwargs(print, sep=" ", flush=True, extra=1) == {"sep": " ", "flush": True}
