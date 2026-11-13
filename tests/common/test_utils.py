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

"""Unit tests for core.common.utils."""

import math
import os
import tempfile
import unittest

from core.common import utils


class TestUtils(unittest.TestCase):
    """Test suite for utility functions in core.common.utils."""

    def test_yaml2dict_scientific_notation(self):
        """Test that scientific notation hyperparameters are parsed as float."""
        yaml_content = """
learning_rate_1: 1e-3
learning_rate_2: 1e-4
learning_rate_3: 1.5e-3
learning_rate_negative: -1e-3
learning_rate_positive: +1e-3
learning_rate_upper: 1E-3
exponent_positive: 1e3
quoted_scientific_double: "1e-3"
quoted_scientific_single: '1e-4'
standard_float: 0.001
infinity_pos: .inf
infinity_neg: -.inf
nan_value: .nan
integer_val: 100
string_val: adam
boolean_val: true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp_file:
            temp_file.write(yaml_content)
            temp_file_path = temp_file.name

        try:
            data = utils.yaml2dict(temp_file_path)

            # Scientific notation parsed as floats
            self.assertIsInstance(data["learning_rate_1"], float)
            self.assertAlmostEqual(data["learning_rate_1"], 0.001)

            self.assertIsInstance(data["learning_rate_2"], float)
            self.assertAlmostEqual(data["learning_rate_2"], 0.0001)

            self.assertIsInstance(data["learning_rate_3"], float)
            self.assertAlmostEqual(data["learning_rate_3"], 0.0015)

            self.assertIsInstance(data["learning_rate_negative"], float)
            self.assertAlmostEqual(data["learning_rate_negative"], -0.001)

            self.assertIsInstance(data["learning_rate_positive"], float)
            self.assertAlmostEqual(data["learning_rate_positive"], 0.001)

            self.assertIsInstance(data["learning_rate_upper"], float)
            self.assertAlmostEqual(data["learning_rate_upper"], 0.001)

            self.assertIsInstance(data["exponent_positive"], float)
            self.assertAlmostEqual(data["exponent_positive"], 1000.0)

            # Quoted scientific notation preserved as strings
            self.assertIsInstance(data["quoted_scientific_double"], str)
            self.assertEqual(data["quoted_scientific_double"], "1e-3")

            self.assertIsInstance(data["quoted_scientific_single"], str)
            self.assertEqual(data["quoted_scientific_single"], "1e-4")

            # Standard types preserved
            self.assertIsInstance(data["standard_float"], float)
            self.assertAlmostEqual(data["standard_float"], 0.001)

            self.assertTrue(math.isinf(data["infinity_pos"]) and data["infinity_pos"] > 0)
            self.assertTrue(math.isinf(data["infinity_neg"]) and data["infinity_neg"] < 0)
            self.assertTrue(math.isnan(data["nan_value"]))

            self.assertIsInstance(data["integer_val"], int)
            self.assertEqual(data["integer_val"], 100)

            self.assertIsInstance(data["string_val"], str)
            self.assertEqual(data["string_val"], "adam")

            self.assertIsInstance(data["boolean_val"], bool)
            self.assertTrue(data["boolean_val"])
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_yaml2dict_invalid_format(self):
        """Test yaml2dict raises RuntimeError for non-yaml files."""
        with self.assertRaises(RuntimeError):
            utils.yaml2dict("config.json")


if __name__ == "__main__":
    unittest.main()
