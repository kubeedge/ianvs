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

import json
import sys
import tempfile
import unittest
from pathlib import Path


VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VALIDATOR_ROOT))

import static_validator  # noqa: E402
import validation_runner  # noqa: E402
from services import report_generator  # noqa: E402
from services.result_contract import RESULT_SCHEMA_VERSION  # noqa: E402


def check(status, name="contract", details=None):
    return static_validator.CheckResult(
        name=name,
        status=status,
        details=list(details or []),
    )


class ResultContractTests(unittest.TestCase):
    def test_json_report_marks_skipped_only_result_as_not_executed_or_passed(self):
        report = validation_runner.skip_dynamic_examples(
            [
                {
                    "name": "inactive",
                    "path": "examples/inactive",
                    "status": "unvalidated",
                }
            ]
        )

        payload = json.loads(static_validator.render_json(report))

        self.assertEqual(RESULT_SCHEMA_VERSION, payload["schema_version"])
        self.assertFalse(payload["passed"])
        self.assertFalse(payload["examples"][0]["passed"])
        self.assertFalse(payload["examples"][0]["executed"])
        self.assertEqual("SKIP", payload["examples"][0]["checks"][0]["status"])

    def test_executed_pass_is_distinct_from_skip(self):
        report = static_validator.StaticValidationReport(
            reports=[
                static_validator.ExampleReport(
                    name="active",
                    path="examples/active",
                    checks=[check("PASS")],
                )
            ]
        )

        payload = json.loads(static_validator.render_json(report))
        self.assertTrue(payload["passed"])
        self.assertTrue(payload["examples"][0]["executed"])
        self.assertTrue(payload["examples"][0]["passed"])

    def test_mixed_status_aggregation_keeps_blocking_precedence(self):
        examples = [
            report_generator.ExampleResult(
                name="mixed",
                path="examples/mixed",
                passed=False,
                checks=[
                    report_generator.CheckResult("pass", "PASS"),
                    report_generator.CheckResult("warn", "WARNING"),
                    report_generator.CheckResult("skip", "SKIP"),
                    report_generator.CheckResult("fail", "FAIL"),
                ],
            )
        ]
        combined = report_generator.CombinedReport(examples, [])

        self.assertFalse(combined.passed)
        self.assertEqual("FAIL", report_generator.dynamic_example_result(examples[0]))
        self.assertEqual("ERROR", report_generator.static_example_result(examples[0]))

    def test_each_result_status_has_stable_display_and_blocking_semantics(self):
        expected = {
            "PASS": ("PASS", "PASS", False),
            "WARNING": ("PASS", "WARNING", False),
            "SKIP": ("SKIP", "SKIP", False),
            "FAIL": ("FAIL", "ERROR", True),
            "ERROR": ("FAIL", "ERROR", True),
        }
        for status, (dynamic, static, blocking) in expected.items():
            with self.subTest(status=status):
                example = report_generator.ExampleResult(
                    "example",
                    "examples/example",
                    not blocking,
                    [report_generator.CheckResult("check", status)],
                )
                self.assertEqual(
                    dynamic,
                    report_generator.dynamic_example_result(example),
                )
                self.assertEqual(
                    static,
                    report_generator.static_example_result(example),
                )
                self.assertEqual(blocking, example.has_blocking_errors)

    def test_duplicate_reports_merge_details_without_duplication(self):
        first = report_generator.ExampleResult(
            "example",
            "examples/example/",
            True,
            [report_generator.CheckResult("check", "WARNING", details=["one"])],
        )
        second = report_generator.ExampleResult(
            "example",
            "examples/example",
            True,
            [report_generator.CheckResult("check", "WARNING", details=["one", "two"])],
        )

        merged = report_generator.merge_duplicate_examples([first, second])

        self.assertEqual(1, len(merged))
        self.assertEqual(["one", "two"], merged[0].checks[0].details)

    def test_missing_result_artifacts_return_contract_error(self):
        self.assertEqual(2, report_generator.main(["--results", "does-not-exist"]))

    def test_malformed_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text("{not-json", encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                report_generator.load_combined_report([path])

    def test_incompatible_schema_version_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(
                json.dumps({"schema_version": 999, "examples": []}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "unsupported.*schema version"):
                report_generator.load_combined_report([path])

    def test_legacy_unversioned_artifact_remains_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text(json.dumps({"examples": []}), encoding="utf-8")

            report = report_generator.load_combined_report([path])
            self.assertEqual([], report.examples)


if __name__ == "__main__":
    unittest.main()
