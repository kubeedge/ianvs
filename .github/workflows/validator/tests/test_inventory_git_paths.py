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

"""Exercise inventory selection using real Git tree paths and the CLI."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


VALIDATOR_ROOT = Path(__file__).resolve().parents[1]


class GitPathSelectionTests(unittest.TestCase):
    """Git's display quoting must not change the selected validation targets."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.git("init", "-q")
        self.git("config", "core.quotePath", "true")
        self.empty_tree = self.git("write-tree").strip()
        self.blob = subprocess.check_output(
            ["git", "hash-object", "-w", "--stdin"],
            cwd=self.root,
            input=b"benchmarkingjob: {}\n",
        ).decode().strip()
        self.inventory = self.root / "inventory.yaml"
        self.inventory.write_text(
            "examples:\n"
            "- name: sample\n"
            "  status: active\n"
            "  path: examples/sample\n"
            "  benchmark_file: examples/sample/benchmarkingjob.yaml\n",
            encoding="utf-8",
        )

    def git(self, *arguments):
        return subprocess.check_output(
            ["git", *arguments], cwd=self.root, text=True
        )

    def changed_tree(self, path):
        self.git("read-tree", self.empty_tree)
        self.git("update-index", "--add", "--cacheinfo", "100644", self.blob, path)
        return self.git("write-tree").strip()

    def detect(self, tree, mode):
        output = self.root / "outputs.txt"
        output.write_text("", encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                str(VALIDATOR_ROOT / "services/inventory_loader.py"),
                "--inventory", str(self.inventory),
                "--base-ref", self.empty_tree,
                "--head-ref", tree,
                "--mode", mode,
                "--github-output", str(output),
            ],
            cwd=self.root,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return dict(line.split("=", 1) for line in output.read_text().splitlines())

    def test_changed_yaml_paths_select_examples(self):
        names = ["variant.yaml", "two words.yaml", "配置.yaml"]
        if os.name != "nt":
            names.extend(['quoted"name.yaml', "back\\slash.yaml", "tab\tname.yaml",
                          "new\nline.yaml", "carriage\rreturn.yaml"])
        for name in names:
            path = "examples/sample/" + name
            tree = self.changed_tree(path)
            for mode in ("static", "dynamic"):
                with self.subTest(path=path, mode=mode):
                    report = self.detect(tree, mode)
                    self.assertEqual(json.loads(report["changed_files"]), [path])
                    self.assertEqual(report["examples_changed"], "true")
                    self.assertEqual(
                        json.loads(report["changed_examples"]),
                        ["examples/sample/benchmarkingjob.yaml"],
                    )

    def test_core_unicode_path_selects_all_dynamic_examples(self):
        path = "core/common/配置.py"
        report = self.detect(self.changed_tree(path), "dynamic")
        self.assertEqual(json.loads(report["changed_files"]), [path])
        self.assertEqual(report["run_all"], "true")
        self.assertEqual(report["examples_changed"], "true")

    def test_unchanged_trees_select_no_examples(self):
        for mode in ("static", "dynamic"):
            with self.subTest(mode=mode):
                report = self.detect(self.empty_tree, mode)
                self.assertEqual(json.loads(report["changed_files"]), [])
                self.assertEqual(report["examples_changed"], "false")


if __name__ == "__main__":
    unittest.main()
