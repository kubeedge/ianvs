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

from services import regression_detector  # noqa: E402


def result_payload(status, details=None, eligibility=False):
    return {
        "schema_version": 1,
        "examples": [
            {
                "name": "example-job",
                "path": "examples/example",
                "checks": [
                    {
                        "name": (
                            "Dynamic validation eligibility"
                            if eligibility
                            else "Runtime smoke test"
                        ),
                        "status": status,
                        "message": "contract result",
                        "file": "examples/example/task.py",
                        "details": list(details or []),
                    }
                ],
            }
        ],
    }


class RegressionContractTests(unittest.TestCase):
    def compare(self, base_payload, head_payload):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.json"
            head = root / "head.json"
            base.write_text(json.dumps(base_payload), encoding="utf-8")
            head.write_text(json.dumps(head_payload), encoding="utf-8")
            return regression_detector.compare_results([base], [head])

    def test_new_failure_is_a_blocking_pr_regression(self):
        report = self.compare(result_payload("PASS"), result_payload("FAIL"))

        self.assertTrue(report.blocks_pr)
        self.assertEqual(
            regression_detector.CLASS_PR_REGRESSION,
            report.comparisons[0].classification,
        )

    def test_unchanged_failure_is_pre_existing_and_nonblocking(self):
        failure = result_payload("FAIL", ["TypeError: existing"])
        report = self.compare(failure, failure)

        self.assertFalse(report.blocks_pr)
        self.assertEqual(
            regression_detector.CLASS_PRE_EXISTING,
            report.comparisons[0].classification,
        )

    def test_fixed_failure_is_classified_as_resolved(self):
        report = self.compare(result_payload("FAIL"), result_payload("PASS"))

        self.assertFalse(report.blocks_pr)
        self.assertEqual(
            regression_detector.CLASS_FIXED_BASELINE,
            report.comparisons[0].classification,
        )

    def test_new_warning_is_reported_but_does_not_block(self):
        report = self.compare(
            result_payload("PASS"),
            result_payload("WARNING", ["new warning"]),
        )

        self.assertFalse(report.blocks_pr)
        self.assertEqual(1, report.new_warning_count)
        self.assertEqual(
            regression_detector.CLASS_PASSED,
            report.comparisons[0].classification,
        )

    def test_skipped_dynamic_unit_does_not_create_regression(self):
        report = self.compare(
            result_payload("PASS"),
            result_payload("SKIP", ["inventory status: unvalidated"], True),
        )

        self.assertFalse(report.blocks_pr)
        self.assertEqual([], report.comparisons)

    def test_added_skipped_unit_is_visible_but_nonblocking(self):
        empty = {"schema_version": 1, "examples": []}
        report = self.compare(
            empty,
            result_payload("SKIP", ["inventory status: unvalidated"], True),
        )

        self.assertFalse(report.blocks_pr)
        self.assertEqual("Added", report.example_changes[0].change)
        self.assertEqual("Skipped", report.example_changes[0].validation)

    def test_incompatible_head_schema_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported.*schema version"):
            self.compare(
                result_payload("PASS"),
                {"schema_version": 2, "examples": []},
            )


if __name__ == "__main__":
    unittest.main()
