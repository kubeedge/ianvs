# Copyright 2022 The KubeEdge Authors.
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

"""
Regression tests for BASE_MODEL_URL scoping in the SingleTaskLearning paradigm.

There is no CI unit-test workflow in this repo yet, so run these manually from
the repository root:

    python -m unittest test.test_singletask_learning_base_model_url

These tests exercise ``_scoped_base_model_url`` directly and through a small
stand-in that mirrors how ``SingleTaskLearning`` uses it (scope wraps both
module construction and training), without pulling in sedna / dataset
machinery.

They lock in the fix for PR #575: a basemodel constructor runs inside
``ParadigmBase.__init__`` - before ``_train`` - so a stale BASE_MODEL_URL left
by a previous test case must not leak into the next one, and a training
exception must not leak the variable either.
"""

import os
import unittest
from contextlib import contextmanager

from core.testcasecontroller.algorithm.paradigm.singletask_learning.singletask_learning \
    import _scoped_base_model_url


ENV = "BASE_MODEL_URL"


@contextmanager
def _clean_env():
    """Restore BASE_MODEL_URL to its original state around a test."""
    sentinel = object()
    original = os.environ.get(ENV, sentinel)
    os.environ.pop(ENV, None)
    try:
        yield
    finally:
        if original is sentinel:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = original


class FakeBaseModel:
    """Records the BASE_MODEL_URL value visible at construction and at train."""

    def __init__(self):
        self.url_at_construction = os.environ.get(ENV)
        self.url_at_train = None

    def train(self, raise_exc=False):
        self.url_at_train = os.environ.get(ENV)
        if raise_exc:
            raise RuntimeError("boom during training")


class FakeSingleTaskLearning:
    """
    Mirror of the SingleTaskLearning construct/run structure that matters here:

      * __init__ builds the basemodel inside the scope (like
        ParadigmBase.__init__ / _get_module_instances)
      * run() re-enters the scope around training
    """

    def __init__(self, initial_model):
        self.initial_model = initial_model
        with _scoped_base_model_url(self.initial_model):
            self.model = FakeBaseModel()

    def run(self, raise_exc=False):
        with _scoped_base_model_url(self.initial_model):
            self.model.train(raise_exc=raise_exc)


class ScopedBaseModelUrlTest(unittest.TestCase):

    def test_sets_value_when_truthy_and_restores_unset(self):
        with _clean_env():
            self.assertNotIn(ENV, os.environ)
            with _scoped_base_model_url("/models/a.pth"):
                self.assertEqual(os.environ[ENV], "/models/a.pth")
            self.assertNotIn(ENV, os.environ)

    def test_empty_string_is_treated_as_no_model(self):
        with _clean_env():
            os.environ[ENV] = "/models/leftover.pth"
            with _scoped_base_model_url(""):
                self.assertNotIn(ENV, os.environ)
            self.assertEqual(os.environ[ENV], "/models/leftover.pth")

    def test_none_is_treated_as_no_model(self):
        with _clean_env():
            os.environ[ENV] = "/models/leftover.pth"
            with _scoped_base_model_url(None):
                self.assertNotIn(ENV, os.environ)
            self.assertEqual(os.environ[ENV], "/models/leftover.pth")

    def test_restores_previous_value_exactly(self):
        with _clean_env():
            os.environ[ENV] = "/models/original.pth"
            with _scoped_base_model_url("/models/temp.pth"):
                self.assertEqual(os.environ[ENV], "/models/temp.pth")
            self.assertEqual(os.environ[ENV], "/models/original.pth")

    def test_restores_on_exception(self):
        with _clean_env():
            os.environ[ENV] = "/models/original.pth"
            with self.assertRaises(RuntimeError):
                with _scoped_base_model_url("/models/temp.pth"):
                    raise RuntimeError("boom")
            self.assertEqual(os.environ[ENV], "/models/original.pth")


class TestCaseIsolationTest(unittest.TestCase):
    """Test case A followed by test case B in the same process."""

    def test_model_then_no_model_does_not_leak_to_constructor(self):
        with _clean_env():
            case_a = FakeSingleTaskLearning("/models/a.pth")
            case_a.run()
            self.assertEqual(case_a.model.url_at_construction, "/models/a.pth")
            self.assertEqual(case_a.model.url_at_train, "/models/a.pth")

            case_b = FakeSingleTaskLearning("")
            case_b.run()
            # B's basemodel constructor must NOT see A's leftover value.
            self.assertIsNone(case_b.model.url_at_construction)
            self.assertIsNone(case_b.model.url_at_train)

    def test_model_x_then_model_y_constructor_sees_y(self):
        with _clean_env():
            case_a = FakeSingleTaskLearning("/models/x.pth")
            case_a.run()

            case_b = FakeSingleTaskLearning("/models/y.pth")
            case_b.run()
            self.assertEqual(case_b.model.url_at_construction, "/models/y.pth")
            self.assertEqual(case_b.model.url_at_train, "/models/y.pth")

    def test_training_exception_in_case_a_does_not_leak_to_case_b(self):
        with _clean_env():
            case_a = FakeSingleTaskLearning("/models/a.pth")
            with self.assertRaises(RuntimeError):
                case_a.run(raise_exc=True)

            self.assertNotIn(ENV, os.environ)

            case_b = FakeSingleTaskLearning("")
            case_b.run()
            self.assertIsNone(case_b.model.url_at_construction)
            self.assertIsNone(case_b.model.url_at_train)

    def test_ambient_value_is_restored_after_each_case(self):
        with _clean_env():
            os.environ[ENV] = "/models/ambient.pth"

            case_a = FakeSingleTaskLearning("/models/a.pth")
            case_a.run()
            self.assertEqual(os.environ[ENV], "/models/ambient.pth")

            case_b = FakeSingleTaskLearning("")
            case_b.run()
            self.assertEqual(os.environ[ENV], "/models/ambient.pth")


if __name__ == "__main__":
    unittest.main()
