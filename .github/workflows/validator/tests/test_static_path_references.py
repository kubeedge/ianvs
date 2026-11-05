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

"""Regression checks for local references and opaque remote URI contents."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from static_validator import ERROR, ExampleReport, _check_repo_path_references


class StaticPathReferenceTests(unittest.TestCase):
    """Remote URI contents must not become blocking checkout-relative paths."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def check_source(self, text, filename="source.py"):
        source = self.root / filename
        source.write_text(text, encoding="utf-8")
        report = ExampleReport(name="uri_probe", path="examples/uri_probe")
        _check_repo_path_references(report, self.root, [source])
        return report

    def test_remote_uri_contents_are_not_local_references(self):
        uris = [
            "https://example.org/examples/quickstart.py",
            "http://example.org/resources/config.yaml",
            "https://example.org/.github/workflows/check.yml",
            "https://example.org/search?path=examples/quickstart.py",
            "https://example.org/page#examples/quickstart.py",
            "s3://bucket/examples/quickstart.py",
            "git+https://example.org/examples/quickstart.py",
        ]
        for uri in uris:
            with self.subTest(uri=uri):
                report = self.check_source("upstream = {!r}\n".format(uri))
                self.assertTrue(report.passed, report.checks)

    def test_remote_documentation_comment_does_not_block(self):
        report = self.check_source(
            "# Upstream documentation: https://example.org/examples/quickstart.py\n"
        )
        self.assertTrue(report.passed, report.checks)

    def test_missing_local_reference_after_url_keeps_line_number(self):
        report = self.check_source(
            'upstream = "https://example.org/examples/quickstart.py"\n'
            "\n"
            'module = "examples/missing.py"\n'
        )
        errors = [check for check in report.checks if check.status == ERROR]
        self.assertEqual(len(errors), 1)
        self.assertEqual(len(errors[0].details), 1)
        self.assertTrue(
            errors[0].details[0].endswith("(Line 3): examples/missing.py"),
            errors[0].details,
        )

    def test_local_reference_on_same_line_as_url_is_checked(self):
        report = self.check_source(
            'upstream = "https://example.org/examples/quickstart.py"; '
            'module = "examples/missing.py"\n'
        )
        errors = [check for check in report.checks if check.status == ERROR]
        self.assertEqual(len(errors), 1)
        self.assertEqual(len(errors[0].details), 1)
        self.assertTrue(
            errors[0].details[0].endswith("(Line 1): examples/missing.py"),
            errors[0].details,
        )

    def test_existing_and_missing_local_references_are_distinct(self):
        local = self.root / "examples" / "local.py"
        local.parent.mkdir()
        local.touch()
        report = self.check_source('module = "./examples/local.py"\n')
        self.assertTrue(report.passed, report.checks)

        report = self.check_source('module = "examples/missing.py"\n')
        self.assertFalse(report.passed)
        self.assertTrue(any(check.status == ERROR for check in report.checks))

    def test_yaml_flow_separators_do_not_hide_local_references(self):
        for text in [
            "references: [https://example.org/docs,examples/missing.py]\n",
            "references: {url: https://example.org/docs,local: examples/missing.py}\n",
        ]:
            with self.subTest(text=text):
                report = self.check_source(text, "config.yaml")
                errors = [check for check in report.checks if check.status == ERROR]
                self.assertEqual(len(errors), 1)
                self.assertEqual(len(errors[0].details), 1)
                self.assertTrue(
                    errors[0].details[0].endswith("(Line 1): examples/missing.py"),
                    errors[0].details,
                )

    def test_quoted_yaml_uri_can_contain_commas(self):
        report = self.check_source(
            'references: ["https://example.org/a,examples/remote.py",'
            '"examples/missing.py"]\n',
            "config.yaml",
        )
        errors = [check for check in report.checks if check.status == ERROR]
        self.assertEqual(len(errors), 1)
        self.assertEqual(len(errors[0].details), 1)
        self.assertTrue(errors[0].details[0].endswith("examples/missing.py"))

    def test_yaml_block_scalar_keeps_local_reference_line(self):
        report = self.check_source(
            "notes: |\n"
            "  https://example.org/examples/quickstart.py\n"
            "  examples/missing.py\n",
            "config.yaml",
        )
        errors = [check for check in report.checks if check.status == ERROR]
        self.assertEqual(len(errors), 1)
        self.assertEqual(len(errors[0].details), 1)
        self.assertTrue(errors[0].details[0].endswith("(Line 3): examples/missing.py"))

    def test_uri_like_suffix_does_not_hide_missing_local_code(self):
        report = self.check_source('module = "examples/missing.py://suffix"\n')
        errors = [check for check in report.checks if check.status == ERROR]
        self.assertEqual(len(errors), 1)
        self.assertEqual(len(errors[0].details), 1)
        self.assertTrue(errors[0].details[0].endswith("examples/missing.py"))


if __name__ == "__main__":
    unittest.main()
