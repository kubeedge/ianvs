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

"""JSONL validation must report damaged data without losing the batch report."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import smoke_test_validator as smoke
from static_validator import render_json


class JsonlDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix="ianvs-jsonl-test-")
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.path = self.root / "test.jsonl"

    def test_invalid_utf8_is_reported_with_line_number(self):
        self.path.write_bytes(b'{"ok": 1}\n{"damaged": "\xff"}\n[]\n')
        issues = smoke._validate_jsonl_file(self.path, self.root)
        self.assertEqual(len(issues), 2)
        self.assertIn("test.jsonl:2:", issues[0])
        self.assertIn("UTF-8", issues[0])
        self.assertIn("test.jsonl:3: row is not a JSON object", issues[1])

    def test_damaged_example_does_not_abort_later_reports(self):
        self.path.write_bytes(b'\xff\n')
        (self.root / "good.jsonl").write_text('{"ok": true}\n', encoding="utf-8")
        examples = [
            {"name": name, "path": "examples/" + name,
             "dataset": {"root": ".", "structure": [filename]}}
            for name, filename in (("bad", "test.jsonl"), ("good", "good.jsonl"))
        ]
        report = smoke.validate_jsonl_examples(self.root, examples)
        serialized = json.loads(render_json(report))
        self.assertFalse(serialized["passed"])
        self.assertEqual(len(serialized["examples"]), 2)
        self.assertFalse(report.reports[0].passed)
        self.assertTrue(report.reports[1].passed)

    def test_read_failure_is_a_diagnostic(self):
        self.path.write_text('{}\n', encoding="utf-8")
        with mock.patch.object(Path, "open", side_effect=PermissionError("access denied")):
            issues = smoke._validate_jsonl_file(self.path, self.root)
        self.assertEqual(len(issues), 1)
        self.assertIn("test.jsonl", issues[0])
        self.assertIn("access denied", issues[0])

    def test_cli_keeps_good_example_result_and_returns_failure(self):
        self.path.write_bytes(b'\xff\n')
        (self.root / "good.jsonl").write_text('{}\n', encoding="utf-8")
        inventory = {"examples": [
            {"name": name, "path": "examples/" + name, "status": "active",
             "dataset": {"root": ".", "structure": [filename]}}
            for name, filename in (("bad", "test.jsonl"), ("good", "good.jsonl"))
        ]}
        # JSON is also valid YAML, the inventory loader's input format.
        (self.root / "inventory.yaml").write_text(json.dumps(inventory), encoding="utf-8")
        runner = Path(smoke.__file__).with_name("validation_runner.py")
        completed = subprocess.run(
            [sys.executable, str(runner), "--jsonl", "--all",
             "--inventory", "inventory.yaml", "--format", "json", "--report", "result.json"],
            cwd=str(self.root), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=10, check=False,
        )
        self.assertEqual(completed.returncode, 1, completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)
        report = json.loads((self.root / "result.json").read_text(encoding="utf-8"))
        self.assertFalse(report["passed"])
        examples = {example["name"]: example for example in report["examples"]}
        self.assertFalse(examples["bad"]["passed"])
        self.assertTrue(examples["good"]["passed"])
        self.assertTrue(any(check["status"] == "FAIL" for check in examples["bad"]["checks"]))

    def test_large_file_is_streamed_and_late_errors_are_found(self):
        with self.path.open("w", encoding="utf-8") as stream:
            for _ in range(20000):
                stream.write('{"text": "valid row"}\n')
            stream.write('[]\n')
        with mock.patch.object(Path, "read_text", side_effect=AssertionError("whole-file read")):
            issues = smoke._validate_jsonl_file(self.path, self.root)
        self.assertEqual(issues, ["test.jsonl:20001: row is not a JSON object"])

    def test_existing_empty_and_row_rules(self):
        for content, allow_empty, expected_count in (
            (b"", False, 1), (b"", True, 0),
            (b"\n", True, 1), (b"{}\n\n[]\nbroken\n", False, 3),
            ('{"text": "中文"}\r\n{}\r\n'.encode("utf-8"), False, 0),
            ('{"text": "a\u2028b"}\n'.encode("utf-8"), False, 0),
        ):
            with self.subTest(content=content, allow_empty=allow_empty):
                self.path.write_bytes(content)
                issues = smoke._validate_jsonl_file(self.path, self.root, allow_empty)
                self.assertEqual(len(issues), expected_count)

    def test_missing_file(self):
        self.assertEqual(smoke._validate_jsonl_file(self.path, self.root),
                         ["test.jsonl: file is missing"])

    def test_diagnostics_are_bounded_but_count_all_invalid_rows(self):
        limit = smoke.MAX_JSONL_ISSUES_PER_FILE
        self.path.write_bytes(b'[]\n' * (limit + 23) + b'{"valid": true}\n')
        issues = smoke._validate_jsonl_file(self.path, self.root)
        self.assertEqual(len(issues), limit + 1)
        self.assertIn("test.jsonl:1:", issues[0])
        self.assertIn("test.jsonl:{}:".format(limit), issues[-2])
        self.assertEqual(issues[-1], "test.jsonl: 23 additional invalid rows omitted")


if __name__ == "__main__":
    unittest.main()
