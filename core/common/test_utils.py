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

import os
import sys
import unittest
import tempfile
from core.common import utils


class TestUtilsSysPathCleanup(unittest.TestCase):
    """Unit tests for sys.path cleanup in utils module."""

    def setUp(self):
        self.original_sys_path = list(sys.path)

    def tearDown(self):
        sys.path = list(self.original_sys_path)

    def test_load_module_sys_path_cleanup_on_error(self):
        """Test load_module cleans up sys.path even when module loading fails."""
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_py = os.path.join(temp_dir, "broken_module.py")
            with open(invalid_py, "w", encoding="utf-8") as f:
                f.write("import non_existent_dependency_12345\n")

            initial_path_len = len(sys.path)
            with self.assertRaises(RuntimeError):
                utils.load_module(invalid_py)

            self.assertEqual(
                len(sys.path),
                initial_path_len,
                "sys.path was not restored after failed load_module",
            )
            self.assertNotIn(
                temp_dir,
                sys.path,
                "Temporary directory leaked into sys.path after load_module failure",
            )

    def test_py2dict_sys_path_cleanup_on_error(self):
        """Test py2dict cleans up sys.path even when py file raises import error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_py = os.path.join(temp_dir, "broken_config.py")
            with open(invalid_py, "w", encoding="utf-8") as f:
                f.write("raise ValueError('Syntax error in config')\n")

            initial_path_len = len(sys.path)
            with self.assertRaises(ValueError):
                utils.py2dict(invalid_py)

            self.assertEqual(
                len(sys.path),
                initial_path_len,
                "sys.path was not restored after failed py2dict",
            )
            self.assertNotIn(
                temp_dir,
                sys.path,
                "Temporary directory leaked into sys.path after py2dict failure",
            )


if __name__ == "__main__":
    unittest.main()
