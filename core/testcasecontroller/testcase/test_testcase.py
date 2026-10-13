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

import os
import tempfile
from core.testcasecontroller.testcase.testcase import TestCase

# Prevent pytest from treating TestCase as a test collection class
TestCase.__test__ = False


class DummyAlgorithm:
    def __init__(self, name):
        self.name = name


def test_get_output_dir_collision():
    algo = DummyAlgorithm("test_algo")
    test_case = TestCase(test_env=None, algorithm=algo)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Pre-create the directory path matching the initial test_case.id
        initial_dir = os.path.join(tmpdir, algo.name, str(test_case.id))
        os.makedirs(initial_dir, exist_ok=True)

        # Call _get_output_dir - should resolve collision
        # by regenerating self.id
        output_dir = test_case._get_output_dir(tmpdir)

        assert output_dir != initial_dir
        assert not os.path.exists(output_dir)
        assert os.path.dirname(output_dir) == os.path.join(tmpdir, algo.name)
