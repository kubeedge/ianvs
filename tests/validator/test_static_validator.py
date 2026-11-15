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
import tempfile
import unittest
from pathlib import Path

# Add validator directory to sys.path
validator_dir = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "validator"
if str(validator_dir) not in sys.path:
    sys.path.insert(0, str(validator_dir))

import static_validator
from static_validator import (
    ERROR,
    PASS,
    ExampleReport,
    _check_repo_path_references,
    _is_inside_spans,
    _remote_url_spans,
)


class TestRemoteUrlSpans(unittest.TestCase):
    def test_http_and_https_urls(self):
        text = "Visit https://host.com/examples/test.py and http://host.com/resources/data.zip"
        spans = _remote_url_spans(text)
        self.assertEqual(len(spans), 2)
        self.assertEqual(text[spans[0][0]:spans[0][1]], "https://host.com/examples/test.py")
        self.assertEqual(text[spans[1][0]:spans[1][1]], "http://host.com/resources/data.zip")

    def test_ftp_and_protocol_relative_urls(self):
        text = "Download from ftp://host/examples/test.py or //cdn.host.com/resources/data.zip"
        spans = _remote_url_spans(text)
        self.assertEqual(len(spans), 2)
        self.assertEqual(text[spans[0][0]:spans[0][1]], "ftp://host/examples/test.py")
        self.assertEqual(text[spans[1][0]:spans[1][1]], "//cdn.host.com/resources/data.zip")

    def test_url_with_query_params_and_fragments(self):
        text = "Check https://example.com/?path=examples/test.py and https://example.com/#examples/test.py"
        spans = _remote_url_spans(text)
        self.assertEqual(len(spans), 2)
        self.assertEqual(text[spans[0][0]:spans[0][1]], "https://example.com/?path=examples/test.py")
        self.assertEqual(text[spans[1][0]:spans[1][1]], "https://example.com/#examples/test.py")

    def test_markdown_links(self):
        text = "See [link](https://github.com/kubeedge/ianvs/blob/main/examples/test.py) for details."
        spans = _remote_url_spans(text)
        self.assertEqual(len(spans), 1)
        self.assertEqual(text[spans[0][0]:spans[0][1]], "https://github.com/kubeedge/ianvs/blob/main/examples/test.py")

    def test_no_urls(self):
        text = "Local file at examples/test.py and ./resources/data.zip"
        spans = _remote_url_spans(text)
        self.assertEqual(spans, [])


class TestIsInsideSpans(unittest.TestCase):
    def test_inside_span(self):
        spans = [(10, 50)]
        self.assertTrue(_is_inside_spans(15, 30, spans))
        self.assertTrue(_is_inside_spans(10, 50, spans))

    def test_outside_span(self):
        spans = [(10, 50)]
        self.assertFalse(_is_inside_spans(0, 5, spans))
        self.assertFalse(_is_inside_spans(50, 60, spans))
        self.assertFalse(_is_inside_spans(5, 20, spans))
        self.assertFalse(_is_inside_spans(40, 60, spans))

    def test_empty_spans(self):
        self.assertFalse(_is_inside_spans(10, 20, []))


class TestCheckRepoPathReferences(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.temp_dir.name)
        # Create an existing local path
        (self.repo_root / "examples").mkdir(parents=True, exist_ok=True)
        (self.repo_root / "examples" / "existing.py").write_text("# existing", encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_remote_url_paths_are_ignored(self):
        source_file = self.repo_root / "examples" / "test_script.py"
        source_file.write_text(
            '# Ref: https://raw.githubusercontent.com/user/repo/main/examples/nonexistent_sample.py\n'
            '# Link: http://example.com/resources/nonexistent_data.zip\n'
            '# Query: https://example.com/?file=examples/nonexistent_query.py\n'
            '# Protocol-relative: //cdn.example.com/examples/nonexistent_cdn.py\n',
            encoding="utf-8",
        )
        report = ExampleReport(name="test_example", path="examples/test")
        _check_repo_path_references(report, self.repo_root, [source_file])

        # None of the remote URL paths should be flagged as missing
        for check in report.checks:
            self.assertEqual(check.status, PASS, f"Unexpected failure in {check.name}: {check.details}")

    def test_genuine_missing_local_path_is_detected(self):
        source_file = self.repo_root / "examples" / "test_script.py"
        source_file.write_text(
            'config_path = "examples/nonexistent_missing_file.py"\n',
            encoding="utf-8",
        )
        report = ExampleReport(name="test_example", path="examples/test")
        _check_repo_path_references(report, self.repo_root, [source_file])

        path_check = next(c for c in report.checks if c.name == "Repository path references exist")
        self.assertEqual(path_check.status, ERROR)
        self.assertTrue(any("examples/nonexistent_missing_file.py" in d for d in path_check.details))

    def test_genuine_existing_local_path_passes(self):
        source_file = self.repo_root / "examples" / "test_script.py"
        source_file.write_text(
            'import_path = "examples/existing.py"\n'
            'relative_path = "./examples/existing.py"\n'
            'path_assign = path=examples/existing.py\n'
            'path_colon = path:examples/existing.py\n'
            'cli_flag = --config=examples/existing.py\n',
            encoding="utf-8",
        )
        report = ExampleReport(name="test_example", path="examples/test")
        _check_repo_path_references(report, self.repo_root, [source_file])

        for check in report.checks:
            self.assertEqual(check.status, PASS, f"Unexpected failure in {check.name}: {check.details}")

    def test_mixed_remote_url_and_missing_local_path(self):
        source_file = self.repo_root / "examples" / "test_script.py"
        source_file.write_text(
            '# Doc: https://github.com/kubeedge/ianvs/blob/main/examples/nonexistent_doc.py\n'
            'local_missing = "examples/truly_missing.py"\n',
            encoding="utf-8",
        )
        report = ExampleReport(name="test_example", path="examples/test")
        _check_repo_path_references(report, self.repo_root, [source_file])

        path_check = next(c for c in report.checks if c.name == "Repository path references exist")
        self.assertEqual(path_check.status, ERROR)
        # Only truly missing local file should be reported, not the URL
        self.assertEqual(len(path_check.details), 1)
        self.assertTrue("examples/truly_missing.py" in path_check.details[0])


if __name__ == "__main__":
    unittest.main()
