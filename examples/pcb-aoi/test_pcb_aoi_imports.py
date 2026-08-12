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
import unittest


class TestPcbAoiSlimImport(unittest.TestCase):
    """Test unit for PCB-AoI tf_slim / tensorflow.contrib.slim compatibility."""

    def test_singletask_basemodel_import_statement(self):
        file_path = os.path.join(
            os.path.dirname(__file__),
            "singletask_learning_bench",
            "fault_detection",
            "testalgorithms",
            "fpn",
            "basemodel.py"
        )
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("import tf_slim as slim", content)
        self.assertIn("import tensorflow.contrib.slim as slim", content)
        self.assertNotIn("\nimport tensorflow.contrib.slim as slim\n", content)

    def test_incremental_basemodel_import_statement(self):
        file_path = os.path.join(
            os.path.dirname(__file__),
            "incremental_learning_bench",
            "fault_detection",
            "testalgorithms",
            "fpn",
            "basemodel.py"
        )
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("import tf_slim as slim", content)
        self.assertIn("import tensorflow.contrib.slim as slim", content)
        self.assertNotIn("\nimport tensorflow.contrib.slim as slim\n", content)

    def test_requirements_contains_tf_slim(self):
        req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
        with open(req_path, "r", encoding="utf-8") as f:
            req_content = f.read()

        self.assertIn("tf-slim", req_content)


if __name__ == "__main__":
    unittest.main()
