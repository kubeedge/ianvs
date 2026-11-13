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

"""Unit tests for core.testenvmanager.dataset."""

import os
import tempfile
import unittest

from core.testenvmanager.dataset.dataset import Dataset


class TestDataset(unittest.TestCase):
    """Test suite for Dataset configuration and processing."""

    def test_csv_index_file_processing(self):
        """Test that CSV index files are processed and retained as valid URLs."""
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as train_f, \
             tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as test_f:
            train_f.write("feature,label\n1.0,0\n")
            test_f.write("feature,label\n2.0,1\n")
            train_path = train_f.name
            test_path = test_f.name

        try:
            ds = Dataset({"train_index": train_path, "test_index": test_path})
            ds.process_dataset()

            self.assertIsNotNone(ds.train_url)
            self.assertEqual(ds.train_url, train_path)
            self.assertIsNotNone(ds.test_url)
            self.assertEqual(ds.test_url, test_path)
        finally:
            os.remove(train_path)
            os.remove(test_path)

    def test_csv_data_file_processing(self):
        """Test that CSV data files are processed and retained as valid URLs."""
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as train_f, \
             tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as test_f:
            train_f.write("feature,label\n1.0,0\n")
            test_f.write("feature,label\n2.0,1\n")
            train_path = train_f.name
            test_path = test_f.name

        try:
            ds = Dataset({"train_data": train_path, "test_data": test_path})
            ds.process_dataset()

            self.assertIsNotNone(ds.train_url)
            self.assertEqual(ds.train_url, train_path)
            self.assertIsNotNone(ds.test_url)
            self.assertEqual(ds.test_url, test_path)
        finally:
            os.remove(train_path)
            os.remove(test_path)

    def test_json_and_jsonl_file_processing(self):
        """Test that JSON index and JSONL data files are properly processed."""
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as index_f, \
             tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as data_f:
            index_f.write('{"images": []}')
            data_f.write('{"prompt": "q", "response": "a"}\n')
            index_path = index_f.name
            data_path = data_f.name

        try:
            ds_index = Dataset({"train_index": index_path, "test_index": index_path})
            ds_index.process_dataset()
            self.assertEqual(ds_index.train_url, index_path)
            self.assertEqual(ds_index.test_url, index_path)

            ds_data = Dataset({"train_data": data_path, "test_data": data_path})
            ds_data.process_dataset()
            self.assertEqual(ds_data.train_url, data_path)
            self.assertEqual(ds_data.test_url, data_path)
        finally:
            os.remove(index_path)
            os.remove(data_path)

    def test_txt_index_file_processing(self):
        """Test that TXT index files with absolute paths are preserved."""
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as train_f, \
             tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as test_f:
            train_f.write(f"{train_f.name} 0\n")
            test_f.write(f"{test_f.name} 1\n")
            train_path = train_f.name
            test_path = test_f.name

        try:
            ds = Dataset({"train_index": train_path, "test_index": test_path})
            ds.process_dataset()
            self.assertEqual(ds.train_url, train_path)
            self.assertEqual(ds.test_url, test_path)
        finally:
            os.remove(train_path)
            os.remove(test_path)

    def test_unsupported_index_format_raises_error(self):
        """Test that an unsupported index format raises ValueError rather than returning None."""
        ds = Dataset({})
        with self.assertRaises(ValueError):
            ds._process_index_file("dataset.parquet")

    def test_unsupported_data_format_raises_error(self):
        """Test that an unsupported data format raises ValueError rather than returning None."""
        ds = Dataset({})
        with self.assertRaises(ValueError):
            ds._process_data_file("dataset.parquet")


if __name__ == "__main__":
    unittest.main()
