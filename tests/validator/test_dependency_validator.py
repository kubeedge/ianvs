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

"""Unit tests for dependency_validator encoding handling and line parsing."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add validator directory to path for imports
VALIDATOR_DIR = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "validator"
if str(VALIDATOR_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATOR_DIR))

from dependency_validator import _read_requirement_lines


class TestDependencyValidatorEncoding(unittest.TestCase):
    """Test suite for requirement file encoding detection and line parsing."""

    def test_utf8_standard_encoding(self):
        """Test reading standard UTF-8 encoded requirement files."""
        content = "numpy>=1.20.0\npandas==1.3.0\n"
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(content.encode("utf-8"))
            file_path = Path(f.name)

        try:
            lines = _read_requirement_lines(file_path)
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0], (1, "numpy>=1.20.0"))
            self.assertEqual(lines[1], (2, "pandas==1.3.0"))
        finally:
            os.remove(file_path)

    def test_utf8_with_bom(self):
        """Test reading UTF-8 encoded requirement files with BOM."""
        content = "torch==1.13.1\ntorchvision==0.14.1\n"
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(content.encode("utf-8-sig"))
            file_path = Path(f.name)

        try:
            lines = _read_requirement_lines(file_path)
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0], (1, "torch==1.13.1"))
            self.assertEqual(lines[1], (2, "torchvision==0.14.1"))
        finally:
            os.remove(file_path)

    def test_utf16_le_bom(self):
        """Test reading UTF-16 Little Endian encoded requirement files with BOM."""
        content = "scikit-learn==1.0.2\nscipy==1.7.3\n"
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"\xff\xfe" + content.encode("utf-16-le"))
            file_path = Path(f.name)

        try:
            lines = _read_requirement_lines(file_path)
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0], (1, "scikit-learn==1.0.2"))
            self.assertEqual(lines[1], (2, "scipy==1.7.3"))
        finally:
            os.remove(file_path)

    def test_utf16_be_bom(self):
        """Test reading UTF-16 Big Endian encoded requirement files with BOM."""
        content = "pillow==9.0.0\nrequests==2.27.1\n"
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"\xfe\xff" + content.encode("utf-16-be"))
            file_path = Path(f.name)

        try:
            lines = _read_requirement_lines(file_path)
            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0], (1, "pillow==9.0.0"))
            self.assertEqual(lines[1], (2, "requests==2.27.1"))
        finally:
            os.remove(file_path)

    def test_crlf_line_endings_and_comments_preservation(self):
        """Test CRLF line endings, blank lines, and comments with accurate line indices."""
        content = "# Header comment\r\n\r\nflask==2.0.1\r\n# Inline comment\r\nwerkzeug==2.0.1\r\n"
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(content.encode("utf-8"))
            file_path = Path(f.name)

        try:
            lines = _read_requirement_lines(file_path)
            self.assertEqual(len(lines), 2)
            # flask is on line 3
            self.assertEqual(lines[0], (3, "flask==2.0.1"))
            # werkzeug is on line 5
            self.assertEqual(lines[1], (5, "werkzeug==2.0.1"))
        finally:
            os.remove(file_path)

    def test_empty_file(self):
        """Test that an empty file returns an empty list."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            file_path = Path(f.name)

        try:
            lines = _read_requirement_lines(file_path)
            self.assertEqual(lines, [])
        finally:
            os.remove(file_path)

    def test_malformed_encoding_raises_error(self):
        """Test that invalid encoding raises UnicodeDecodeError without silent corruption."""
        invalid_bytes = b"\x80\x81\x82\x83\xff\xfe_not_at_start"
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(invalid_bytes)
            file_path = Path(f.name)

        try:
            with self.assertRaises(UnicodeDecodeError):
                _read_requirement_lines(file_path)
        finally:
            os.remove(file_path)


if __name__ == "__main__":
    unittest.main()
