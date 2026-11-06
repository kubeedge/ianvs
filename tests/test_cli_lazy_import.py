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

"""Tests for the informational CLI flags of ianvs.

`ianvs -v` and `ianvs --help` are argparse actions that exit during
parse_args(). They must therefore never import core.cmd.obj, which pulls in the
full benchmarking stack (sedna, onnx, matplotlib). These tests run the CLI in a
subprocess and assert both the user visible behaviour and the absence of that
import, so that a future top-level import cannot silently restore the
dependency.
"""

import ast
import json
import os
import subprocess
import sys
import unittest

from core.__version__ import __version__

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_WRAPPER = os.path.join(REPO_ROOT, "benchmarking.py")

# Drives the console_scripts entry point, then reports the exit status and
# whether the heavy module was pulled in. The report goes to stderr so that
# stdout stays exactly what a user would see.
_ENTRY_POINT_RUNNER = """
import json, sys
sys.argv = ["ianvs", "{flag}"]
exit_code = 0
try:
    from core.cmd.benchmarking import main
    main()
except SystemExit as exc:
    exit_code = exc.code if isinstance(exc.code, int) else 0
sys.stderr.write(json.dumps({{
    "exit_code": exit_code,
    "obj_imported": "core.cmd.obj" in sys.modules,
}}))
"""

# Same probe for the root-level benchmarking.py, which is a second supported
# way to start ianvs.
_ROOT_WRAPPER_RUNNER = """
import json, runpy, sys
sys.argv = ["ianvs", "{flag}"]
exit_code = 0
try:
    runpy.run_path({wrapper!r}, run_name="__main__")
except SystemExit as exc:
    exit_code = exc.code if isinstance(exc.code, int) else 0
sys.stderr.write(json.dumps({{
    "exit_code": exit_code,
    "obj_imported": "core.cmd.obj" in sys.modules,
}}))
"""


def _run(source):
    """Run `source` in a subprocess and return (stdout, decoded stderr report)."""
    proc = subprocess.run(
        [sys.executable, "-c", source],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=False,
    )
    return proc.stdout, json.loads(proc.stderr)


def _run_entry_point(flag):
    """Run the ianvs console entry point with `flag`."""
    return _run(_ENTRY_POINT_RUNNER.format(flag=flag))


def _run_root_wrapper(flag):
    """Run the root-level benchmarking.py with `flag`."""
    return _run(_ROOT_WRAPPER_RUNNER.format(flag=flag, wrapper=ROOT_WRAPPER))


def _root_wrapper_is_standalone_copy():
    """True while benchmarking.py still imports core.cmd.obj at module level.

    The root file is currently a near duplicate of core/cmd/benchmarking.py
    rather than a wrapper around it, so the deferral in this PR does not reach
    it. PR #672 replaces it with a wrapper that delegates to the fixed core
    command; once that lands, this returns False and the root path is guarded
    by the same assertions as the entry point.
    """
    if not os.path.isfile(ROOT_WRAPPER):
        return False
    with open(ROOT_WRAPPER, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "core.cmd.obj":
            return True
    return False


_ROOT_SKIP_REASON = (
    "root benchmarking.py still imports core.cmd.obj at module level; it is a "
    "standalone copy of the CLI rather than a wrapper. PR #672 makes it "
    "delegate to core.cmd.benchmarking, and this test starts guarding that "
    "path automatically once it lands."
)


class TestCliInformationalFlags(unittest.TestCase):
    """`-v` and `--help` must succeed without importing the benchmarking stack."""

    def _assert_clean(self, stdout, report, expected, label):
        self.assertEqual(report["exit_code"], 0)
        self.assertIn(expected, stdout)
        self.assertFalse(
            report["obj_imported"],
            f"`{label}` imported core.cmd.obj; the import in "
            "core/cmd/benchmarking.py must stay inside main() so informational "
            "flags do not load sedna, onnx and matplotlib.",
        )

    def test_version_flag_exits_cleanly_without_heavy_import(self):
        """`ianvs -v` prints the version, exits 0, and skips core.cmd.obj."""
        stdout, report = _run_entry_point("-v")
        self._assert_clean(stdout, report, __version__, "ianvs -v")

    def test_help_flag_exits_cleanly_without_heavy_import(self):
        """`ianvs --help` prints usage, exits 0, and skips core.cmd.obj."""
        stdout, report = _run_entry_point("--help")
        self._assert_clean(stdout, report, "usage: ianvs", "ianvs --help")

    @unittest.skipIf(_root_wrapper_is_standalone_copy(), _ROOT_SKIP_REASON)
    def test_root_wrapper_version_flag_without_heavy_import(self):
        """The root benchmarking.py -v must also skip core.cmd.obj."""
        stdout, report = _run_root_wrapper("-v")
        self._assert_clean(stdout, report, __version__, "python benchmarking.py -v")

    @unittest.skipIf(_root_wrapper_is_standalone_copy(), _ROOT_SKIP_REASON)
    def test_root_wrapper_help_flag_without_heavy_import(self):
        """The root benchmarking.py --help must also skip core.cmd.obj."""
        stdout, report = _run_root_wrapper("--help")
        self._assert_clean(stdout, report, "usage: ianvs", "python benchmarking.py --help")


if __name__ == "__main__":
    unittest.main()
