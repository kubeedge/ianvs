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

"""Tests for JSONL dataset normalization utilities."""

import json
import tempfile
import unittest
from pathlib import Path

from core.testenvmanager.dataset.utils import rename_keys_jsonl


class TestRenameKeysJsonl(unittest.TestCase):
    def test_normalization_preserves_source_file(self):
        """Normalization should not overwrite the caller's source dataset."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "dataset.jsonl"
            source_content = (
                '{"instruction": "What is edge AI?", '
                '"response": "AI running near the data source."}\n'
            )
            source_path.write_text(source_content, encoding="utf-8")

            normalized_path = rename_keys_jsonl(str(source_path))

            self.assertEqual(
                source_path.read_text(encoding="utf-8"),
                source_content,
            )
            self.assertIsNotNone(normalized_path)
            self.assertNotEqual(Path(normalized_path), source_path)

            normalized_record = json.loads(
                Path(normalized_path).read_text(encoding="utf-8")
            )
            self.assertEqual(
                normalized_record,
                {
                    "question": "What is edge AI?",
                    "answer": "AI running near the data source.",
                },
            )

    def test_already_normalized_file_returns_source_path(self):
        """An already normalized dataset should be reused unchanged."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "dataset.jsonl"
            source_content = (
                '{"question": "What is edge AI?", '
                '"answer": "AI running near the data source."}\n'
            )
            source_path.write_text(source_content, encoding="utf-8")

            returned_path = rename_keys_jsonl(str(source_path))

            self.assertEqual(returned_path, str(source_path))
            self.assertEqual(
                source_path.read_text(encoding="utf-8"),
                source_content,
            )


if __name__ == "__main__":
    unittest.main()
