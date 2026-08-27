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

"""
Regression test for the kind auto-install version pin.

check_host_kind()'s auto-install fallback used to hardcode a specific kind
release (v0.17.0, from ~January 2023) directly inside the install shell
command. This test guards two things: that the install command is built
from the KIND_VERSION module constant rather than a re-hardcoded literal,
and that the constant itself isn't left pointing at that known-stale
release. It does not (and cannot, without a real host) verify the install
actually succeeds -- it only verifies the version wiring stays correct.
"""

import unittest
from unittest.mock import MagicMock, patch

from core.testcasecontroller.simulation_system_admin import simulation_system_admin as ssa


class TestKindVersionPin(unittest.TestCase):
    """Verify the kind install command tracks KIND_VERSION, not a literal."""

    def test_kind_version_is_not_the_known_stale_pin(self):
        """v0.17.0 predates several stable Kubernetes releases; guard against
        reverting to it (or any hardcoded literal) by accident."""
        self.assertNotEqual(ssa.KIND_VERSION, "v0.17.0")
        self.assertRegex(ssa.KIND_VERSION, r"^v\d+\.\d+\.\d+$")

    @patch("core.testcasecontroller.simulation_system_admin.simulation_system_admin.subprocess.run")
    def test_install_command_uses_the_constant(self, mock_run):
        """When kind isn't found, the install command it shells out to must
        embed the current KIND_VERSION constant, not a stale copy of it."""
        # First call (`kind version`) fails -> triggers the install branch.
        # Second call (the install itself) "succeeds".
        mock_run.side_effect = [
            MagicMock(returncode=1),
            MagicMock(returncode=0),
        ]

        ssa.check_host_kind()

        install_call = mock_run.call_args_list[1]
        install_cmd = install_call.args[0]
        self.assertIn(ssa.KIND_VERSION, install_cmd)
        self.assertIn(f"/dl/{ssa.KIND_VERSION}/kind-linux-amd64", install_cmd)


if __name__ == "__main__":
    unittest.main()
