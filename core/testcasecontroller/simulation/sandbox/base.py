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

"""Backend interface every sandbox tier implements."""

from abc import ABC, abstractmethod


class Sandbox(ABC):
    """A backend that runs one test case in isolation and reports metrics."""

    def __init__(self, config, ianvs_root):
        self.config = config
        self.ianvs_root = ianvs_root

    def prepare(self):
        """Provision anything shared across test cases. Optional to override."""

    @abstractmethod
    def run_testcase(self, testcase, workspace):
        """
        Run one test case inside the sandbox.

        Returns
        -------
        (result, profile) : tuple
            ``result`` is the test case's own metrics dict (``None`` on
            failure); ``profile`` is a ``profiler.ProfileResult``.
        """

    def teardown(self):
        """Release anything provisioned by prepare(). Optional to override."""
