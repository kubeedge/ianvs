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

import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock


VALIDATOR_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(VALIDATOR_ROOT))

import smoke_test_validator  # noqa: E402
import static_validator  # noqa: E402


class RuntimeLabelContractTests(unittest.TestCase):
    def run_smoke(self, mocked):
        report = static_validator.ExampleReport("example", "examples/example")
        completed = subprocess.CompletedProcess(["example"], 0, stdout="done")
        with mock.patch.object(
            smoke_test_validator,
            "_configure_mock_runtime",
            return_value=(mocked, ""),
        ), mock.patch.object(
            smoke_test_validator.subprocess,
            "run",
            return_value=completed,
        ):
            smoke_test_validator._run_smoke_command(
                report=report,
                repo_root=REPOSITORY_ROOT,
                example={},
                benchmark_file="examples/example/benchmarkingjob.yaml",
                prepared_dataset_root=None,
                smoke_command=["example"],
                timeout_seconds=1,
            )
        return report.checks[0]

    def test_mocked_runtime_is_explicitly_labeled(self):
        result = self.run_smoke(mocked=True)
        self.assertEqual("Runtime smoke test (mocked_llm)", result.name)
        self.assertIn("substituted LLM responses", result.message)

    def test_real_runtime_does_not_receive_mock_label(self):
        result = self.run_smoke(mocked=False)
        self.assertEqual("Runtime smoke test", result.name)
        self.assertNotIn("substituted", result.message)


class WorkflowContractTests(unittest.TestCase):
    def workflow(self, name):
        return (REPOSITORY_ROOT / ".github" / "workflows" / name).read_text(
            encoding="utf-8"
        )

    def test_static_base_and_head_use_isolated_revisions(self):
        workflow = self.workflow("static_code_requirement_cicd.yaml")
        self.assertIn('git worktree add --detach "${base_dir}" FETCH_HEAD', workflow)
        self.assertIn('cd "${IANVS_BASE_DIR}"', workflow)
        self.assertIn("static-validation-base-${safe_name}", workflow)
        self.assertIn("static-validation-pr-${safe_name}", workflow)

    def test_dynamic_base_and_head_use_distinct_artifact_names(self):
        workflow = self.workflow("dynamic_code_cicd.yaml")
        self.assertIn('cd "${IANVS_BASE_DIR}"', workflow)
        self.assertIn('"${IANVS_BASE_DIR}/.github/workflows/validator/', workflow)
        self.assertIn("dynamic-validation-base-${safe_name}", workflow)
        self.assertIn("dynamic-validation-pr-${safe_name}", workflow)
        self.assertIn("pattern: dynamic-validation-base-*", workflow)
        self.assertIn("pattern: dynamic-validation-pr-*", workflow)

    def test_contract_suite_workflow_has_no_external_runtime_requirements(self):
        workflow = self.workflow("validator_contract_tests.yaml")
        self.assertIn("python -m unittest discover", workflow)
        for forbidden in ("model download", "dataset download", "gpu", "api key"):
            self.assertNotIn(forbidden, workflow.lower())


if __name__ == "__main__":
    unittest.main()
