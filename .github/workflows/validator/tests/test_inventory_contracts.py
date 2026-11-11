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

import sys
import unittest
from pathlib import Path
from unittest import mock


VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VALIDATOR_ROOT))

from services import inventory_loader  # noqa: E402
import validation_runner  # noqa: E402


class InventoryContractTests(unittest.TestCase):
    def setUp(self):
        self.inventory = {
            "examples": [
                {
                    "name": "active-example",
                    "path": "examples/active/",
                    "status": "active",
                    "python_version": "3.9",
                },
                {
                    "name": "inactive-example",
                    "path": "examples/inactive",
                    "status": "unvalidated",
                },
                {"name": "missing-path", "status": "active"},
            ]
        }

    def test_active_selection_excludes_inactive_and_incomplete_entries(self):
        examples = inventory_loader.inventory_examples(self.inventory)

        self.assertEqual(["active-example"], [item["name"] for item in examples])
        self.assertEqual("examples/active", examples[0]["path"])

    def test_explicit_selection_can_include_inactive_example(self):
        examples = inventory_loader.inventory_examples(
            self.inventory,
            active_only=False,
        )

        selected = validation_runner.select_examples(
            examples,
            ["examples/inactive/"],
            include_all=False,
        )
        self.assertEqual(["inactive-example"], [item["name"] for item in selected])

    def test_windows_style_repository_paths_are_normalized(self):
        example = inventory_loader.normalize_example(
            {"name": "windows", "path": "examples\\robotics\\"}
        )

        self.assertEqual("examples/robotics", example["path"])
        self.assertEqual(
            "examples/robotics",
            validation_runner.normalize_selector(".\\examples\\robotics\\"),
        )

    def test_changed_file_matches_only_the_target_path_boundary(self):
        examples = inventory_loader.inventory_examples(self.inventory)

        selected = inventory_loader.detect_static_examples(
            ["examples/active/task.py", "examples/active_extra/task.py"],
            examples,
        )
        self.assertEqual(["active-example"], [item["name"] for item in selected])

    def test_core_or_validator_change_runs_all_dynamic_targets(self):
        self.assertTrue(inventory_loader.should_run_all_dynamic(["core/test.py"]))
        self.assertTrue(
            inventory_loader.should_run_all_dynamic(
                [".github/workflows/validator/static_validator.py"]
            )
        )
        self.assertFalse(
            inventory_loader.should_run_all_dynamic(["docs/example_validator.md"])
        )

    def test_selection_report_preserves_selector_and_python_contract(self):
        examples = inventory_loader.inventory_examples(
            self.inventory,
            active_only=False,
        )

        report = inventory_loader.inventory_selection_report(
            mode="dynamic",
            examples=examples,
            run_all=False,
            changed_files=["examples/active/task.py"],
        )

        self.assertEqual(
            [
                {"selector": "active-example", "python_version": "3.9"},
                {"selector": "inactive-example", "python_version": "3.8"},
            ],
            report["validation_matrix"],
        )
        self.assertTrue(report["examples_changed"])

    def test_empty_selection_returns_a_nonzero_runner_result(self):
        with mock.patch.object(
            validation_runner,
            "load_selected_examples",
            return_value=[],
        ):
            self.assertEqual(1, validation_runner.main(["--static"]))


if __name__ == "__main__":
    unittest.main()
