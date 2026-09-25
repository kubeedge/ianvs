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
Regression tests for core/testcasecontroller/simulation_system_admin.

These target the 5 breakages identified while restoring the 2022 simulation
proposal (Ianvs PR #35 / #39) for the LFX "KubeEdge-Ianvs Simulation Sandbox"
Term 3 2026 project (upstream issue kubeedge/ianvs#348):

1. check_host_docker() silently reported success even when Docker was
   unreachable, because `docker version | head -n 2` made `check=True`
   validate head's exit code instead of docker's.
2. check_host_kind() crashed with CalledProcessError instead of running its
   own auto-install fallback, for the same root cause in the opposite
   direction.
3. get_host_number_of_cpus() crashed on modern util-linux, whose `lscpu`
   emits `CPU(s):` rather than the legacy `CPU:`.
4. Both get_host_number_of_cpus() and get_host_free_memory_size() parsed
   `str(bytes)` (the repr, e.g. "b'CPU(s):  4\\n'") instead of decoding the
   subprocess output as text.
5. build_simulation_enviroment() and destory_simulation_enviroment() pulled
   the sedna installer script from different branches (master vs. main).
"""

import subprocess
import unittest
from unittest.mock import MagicMock, patch

from core.testcasecontroller.simulation_system_admin import simulation_system_admin as ssa

_MODULE = "core.testcasecontroller.simulation_system_admin.simulation_system_admin"


def _popen_returning(stdout_bytes):
    """Build a mock context manager standing in for subprocess.Popen."""
    mock_proc = MagicMock()
    mock_proc.stdout.read.return_value = stdout_bytes
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_proc
    mock_cm.__exit__.return_value = False
    return mock_cm


class TestCheckHostDocker(unittest.TestCase):
    """Bug 1: docker check must actually observe docker's exit code."""

    @patch(f"{_MODULE}.subprocess.run")
    def test_docker_available_skips_install(self, mock_run):
        """Docker present: only the availability check should run."""
        mock_run.return_value = MagicMock(returncode=0)
        ssa.check_host_docker()
        # only the availability check should run; no install attempt
        mock_run.assert_called_once()
        called_args = mock_run.call_args[0][0]
        self.assertEqual(called_args, ["docker", "version"])

    @patch(f"{_MODULE}.subprocess.run")
    def test_docker_unavailable_triggers_install(self, mock_run):
        """Docker absent: the install fallback must actually execute."""
        # first call: availability check fails; second call: install succeeds
        mock_run.side_effect = [
            MagicMock(returncode=1),
            MagicMock(returncode=0),
        ]
        ssa.check_host_docker()
        self.assertEqual(mock_run.call_count, 2)

    @patch(f"{_MODULE}.subprocess.run")
    def test_docker_binary_missing_triggers_install(self, mock_run):
        """Docker binary missing entirely: still falls through to install."""
        mock_run.side_effect = [
            FileNotFoundError(),
            MagicMock(returncode=0),
        ]
        ssa.check_host_docker()
        self.assertEqual(mock_run.call_count, 2)


class TestCheckHostKind(unittest.TestCase):
    """Bug 2: kind check must not crash before its fallback can run."""

    @patch(f"{_MODULE}.subprocess.run")
    def test_kind_available_skips_install(self, mock_run):
        """Kind present: only the availability check should run."""
        mock_run.return_value = MagicMock(returncode=0)
        ssa.check_host_kind()
        mock_run.assert_called_once()
        called_args = mock_run.call_args[0][0]
        self.assertEqual(called_args, ["kind", "version"])

    @patch(f"{_MODULE}.subprocess.run")
    def test_kind_unavailable_does_not_raise_and_installs(self, mock_run):
        """Kind absent: must not raise, and must reach the install fallback."""
        # Previously: check=True on a nonzero-exit "kind version" raised
        # CalledProcessError here, so the install fallback never executed.
        mock_run.side_effect = [
            MagicMock(returncode=127),
            MagicMock(returncode=0),
        ]
        try:
            ssa.check_host_kind()
        except subprocess.CalledProcessError:
            self.fail("check_host_kind() raised instead of running its install fallback")
        self.assertEqual(mock_run.call_count, 2)


class TestGetHostNumberOfCpus(unittest.TestCase):
    """Bugs 3 & 4: modern lscpu label, NUMA-line anchoring, and byte decoding."""

    @patch(f"{_MODULE}.subprocess.Popen")
    def test_parses_modern_lscpu_label(self, mock_popen):
        """Modern util-linux emits CPU(s):, not the legacy CPU:."""
        # Modern util-linux: "CPU(s):", not the legacy "CPU:".
        mock_popen.return_value = _popen_returning(b"CPU(s):                  8\n")
        self.assertEqual(ssa.get_host_number_of_cpus(), 8)

    @patch(f"{_MODULE}.subprocess.Popen")
    def test_does_not_crash_on_empty_output(self, mock_popen):
        """Empty grep match raises a clear error instead of IndexError."""
        # Old behaviour: legacy "CPU:" grep against modern lscpu output
        # returns nothing, and indexing into it raised IndexError.
        mock_popen.return_value = _popen_returning(b"")
        with self.assertRaises(RuntimeError):
            ssa.get_host_number_of_cpus()

    @patch(f"{_MODULE}.subprocess.Popen")
    def test_grep_anchor_used_to_avoid_numa_line(self, mock_popen):
        """The grep pattern must anchor so it cannot match a NUMA line."""
        # The shell command itself must anchor to '^CPU(s):' so it can't
        # match a NUMA node line (whose value is a range, not a count).
        mock_popen.return_value = _popen_returning(b"CPU(s):                  4\n")
        ssa.get_host_number_of_cpus()
        shell_cmd = mock_popen.call_args[0][0]
        self.assertIn("^CPU(s):", shell_cmd)


class TestGetHostFreeMemorySize(unittest.TestCase):
    """Bug 4: memory parser must decode bytes rather than reading repr(bytes)."""

    @patch(f"{_MODULE}.subprocess.Popen")
    def test_parses_decoded_meminfo_line(self, mock_popen):
        """Decoded bytes parse correctly instead of reading repr(bytes)."""
        mock_popen.return_value = _popen_returning(b"MemFree:        3485688 kB\n")
        self.assertEqual(ssa.get_host_free_memory_size(), 3485688)


class TestSednaInstallerUrls(unittest.TestCase):
    """Bug 5: build and teardown must fetch the installer from the same branch."""

    @patch(f"{_MODULE}.subprocess.run")
    @patch(f"{_MODULE}.check_host_enviroment")
    def test_build_uses_shared_constant(self, _mock_check_env, mock_run):
        """Build path pulls the sedna installer from the shared constant."""
        mock_run.return_value = MagicMock(returncode=0)
        simulation = MagicMock(
            cloud_number=1, edge_number=1, kubeedge_version="latest",
            sedna_version="latest", cluster_name="test-cluster",
        )
        ssa.build_simulation_enviroment(simulation)
        shell_cmd = mock_run.call_args[0][0]
        self.assertIn(ssa.SEDNA_INSTALL_SCRIPT_URL, shell_cmd)

    @patch(f"{_MODULE}.subprocess.call")
    def test_destroy_uses_shared_constant(self, mock_call):
        """Teardown path pulls the sedna installer from the shared constant."""
        mock_call.return_value = 0
        simulation = MagicMock(cluster_name="test-cluster")
        ssa.destory_simulation_enviroment(simulation)
        shell_cmd = mock_call.call_args[0][0]
        self.assertIn(ssa.SEDNA_INSTALL_SCRIPT_URL, shell_cmd)

    def test_only_one_url_defined_in_module(self):
        """Sanity-check the constant itself points at the main branch."""
        # Guards against a future contributor reintroducing a second,
        # independently-edited copy of this URL.
        self.assertTrue(ssa.SEDNA_INSTALL_SCRIPT_URL.endswith("all-in-one.sh"))
        self.assertIn("/main/", ssa.SEDNA_INSTALL_SCRIPT_URL)


if __name__ == "__main__":
    unittest.main()
