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

"""Exercise validator timeouts with real, short-lived subprocess trees."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import smoke_test_validator as smoke
from static_validator import ExampleReport, FAIL, PASS
from services.process_runner import run_command


class SmokeTimeoutTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="ianvs-timeout-test-")
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.report = ExampleReport(name="fixture", path="examples/fixture")

    def worker_script(self, parent_exits=False):
        """A leaked worker leaves a marker, then exits without external cleanup."""
        marker = self.root / "worker-survived"
        child = (
            "import time; from pathlib import Path; "
            "time.sleep(2); Path({!r}).write_text('still running')".format(str(marker))
        )
        script = self.root / "prepare.py"
        script.write_text(
            "import subprocess, sys, time\n"
            "subprocess.Popen([sys.executable, '-c', {!r}])\n"
            "print('worker started', flush=True)\n"
            "{}\n".format(child, "" if parent_exits else "time.sleep(20)"),
            encoding="utf-8",
        )
        return script, marker

    def assert_worker_stopped(self, marker):
        # Wait beyond the worker's delayed write. os.kill(pid, 0) alone would
        # mistake a terminated but not yet reaped child for a live worker.
        time.sleep(2.1)
        self.assertFalse(marker.exists(), "worker continued after validator timeout")
        check = self.report.checks[-1]
        self.assertEqual(check.status, FAIL)
        self.assertIn("timed out", check.message)
        self.assertIn("worker started", "\n".join(check.details))

    @unittest.skipUnless(os.name == "posix", "process-group cleanup requires POSIX")
    def test_smoke_timeout_stops_descendant_and_keeps_output(self):
        script, marker = self.worker_script()
        smoke._run_smoke_command(
            self.report, self.root, {}, "fixture.yaml", None,
            [sys.executable, str(script)], 0.4,
        )
        self.assert_worker_stopped(marker)

    @unittest.skipUnless(os.name == "posix", "process-group cleanup requires POSIX")
    def test_timeout_stops_descendant_after_group_leader_exits(self):
        script, marker = self.worker_script(parent_exits=True)
        smoke._run_smoke_command(
            self.report, self.root, {}, "fixture.yaml", None,
            [sys.executable, str(script)], 0.4,
        )
        self.assert_worker_stopped(marker)

    @unittest.skipUnless(os.name == "posix", "process-group cleanup requires POSIX")
    def test_preparation_timeout_stops_descendant_and_keeps_output(self):
        script, marker = self.worker_script()
        success = smoke._run_preparation_step(
            self.report, self.root, self.root,
            {"name": "fixture", "type": "dataset", "script": script.name,
             "args": [], "timeout": 1},
        )
        self.assertFalse(success)
        self.assert_worker_stopped(marker)

    @unittest.skipUnless(os.name == "posix", "process-group cleanup requires POSIX")
    def test_legacy_dataset_timeout_stops_descendant_and_keeps_output(self):
        script, marker = self.worker_script()
        result = smoke._prepare_dataset(
            self.report, self.root,
            {"dataset": {"prepare_script": script.name}},
            self.root / "dataset", 0.4,
        )
        self.assertIsNone(result)
        self.assert_worker_stopped(marker)

    def test_success_and_failure_preserve_exit_status_and_output(self):
        for exit_code in (0, 7):
            with self.subTest(exit_code=exit_code):
                smoke._run_smoke_command(
                    self.report, self.root, {}, "fixture.yaml", None,
                    [sys.executable, "-c", "print('fixture output'); exit({})".format(exit_code)],
                    5,
                )
                check = self.report.checks[-1]
                self.assertEqual(check.status, PASS if exit_code == 0 else FAIL)
                self.assertIn("fixture output", check.details)
                if exit_code:
                    self.assertIn("exit code 7", check.message)

    def test_command_receives_working_directory_and_environment(self):
        completed = run_command(
            [sys.executable, "-c",
             "import os; print(os.getcwd()); print(os.environ['IANVS_TEST_VALUE'])"],
            cwd=str(self.root), env=dict(os.environ, IANVS_TEST_VALUE="fixture"), timeout=5,
        )
        self.assertEqual(completed.returncode, 0)
        self.assertEqual(completed.stdout.splitlines(), [str(self.root.resolve()), "fixture"])

    @unittest.skipUnless(os.name == "posix", "detached sessions require POSIX")
    def test_detached_child_cannot_hold_timeout_output_pipe_open(self):
        # Detached processes are outside the cleanup group. This fixture exits
        # by itself; its inherited pipe must not extend the timeout indefinitely.
        code = (
            "import subprocess, sys, time\n"
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(3)'], "
            "start_new_session=True)\n"
            "print('before timeout', flush=True)\n"
            "time.sleep(20)\n"
        )
        start = time.monotonic()
        try:
            with self.assertRaises(subprocess.TimeoutExpired) as caught:
                run_command([sys.executable, "-c", code], cwd=str(self.root), timeout=0.4)
            elapsed = time.monotonic() - start
            self.assertLess(elapsed, 2.8, "timeout blocked while draining a detached child's pipe")
            self.assertIn("before timeout", caught.exception.stdout)
        finally:
            # Allow the intentionally detached fixture to finish before the
            # test releases its temporary directory.
            time.sleep(max(0, 3.5 - (time.monotonic() - start)))


if __name__ == "__main__":
    unittest.main()
