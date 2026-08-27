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

"""Unit tests for algorithm module hyperparameter parsing."""

import copy
import unittest

from core.testcasecontroller.algorithm.module.module import Module


class TestModuleHyperparameters(unittest.TestCase):
    """Tests for Module hyperparameter configuration parsing."""

    def test_parsing_preserves_input_configuration(self):
        """Parsing should generate combinations without modifying its input."""
        config = {
            "type": "basemodel",
            "name": "DemoModel",
            "url": "",
            "hyperparameters": [
                {"learning_rate": {"values": [0.1, 0.01]}},
                {"batch_size": {"values": [16, 32]}},
            ],
        }
        config_before_parsing = copy.deepcopy(config)

        module = Module(config)

        expected_combinations = [
            {"learning_rate": 0.1, "batch_size": 16},
            {"learning_rate": 0.1, "batch_size": 32},
            {"learning_rate": 0.01, "batch_size": 16},
            {"learning_rate": 0.01, "batch_size": 32},
        ]

        self.assertEqual(module.hyperparameters_list, expected_combinations)
        self.assertEqual(config, config_before_parsing)

    def _base_config(self, hyperparameters):
        return {
            "type": "basemodel",
            "name": "DemoModel",
            "url": "",
            "hyperparameters": hyperparameters,
        }

    def test_multi_key_entry_raises(self):
        """An entry defining two hyperparameters should not be silently truncated."""
        config = self._base_config([
            {"learning_rate": {"values": [0.1]}, "batch_size": {"values": [16]}},
        ])
        with self.assertRaises(ValueError):
            Module(config)

    def test_empty_entry_raises(self):
        """An entry defining no hyperparameter should be rejected."""
        with self.assertRaises(ValueError):
            Module(self._base_config([{}]))

    def test_non_mapping_entry_raises(self):
        """A non-dict entry should be rejected with a clear error."""
        with self.assertRaises(ValueError):
            Module(self._base_config(["learning_rate"]))

    def test_non_mapping_hyperparameter_config_raises(self):
        """A hyperparameter whose config is not a dict should be rejected."""
        with self.assertRaises(ValueError):
            Module(self._base_config([{"learning_rate": [0.1, 0.01]}]))


if __name__ == "__main__":
    unittest.main()
