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

import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.testenvmanager.dataset import Dataset


def test_split_dataset_parameter_validation():
    ds = Dataset({})

    # Invalid times parameter (times < 1)
    with pytest.raises(ValueError, match="times"):
        ds.split_dataset("sample.jsonl", "jsonl", ratio=0.8, times=0)

    # Invalid ratio parameter (ratio >= 1.0 or <= 0.0)
    with pytest.raises(ValueError, match="ratio"):
        ds.split_dataset("sample.jsonl", "jsonl", ratio=1.5, times=2)

    with pytest.raises(ValueError, match="ratio"):
        ds.split_dataset("sample.jsonl", "jsonl", ratio=0.0, times=2)

    # Invalid times for city_splitting (times < 2)
    with pytest.raises(ValueError, match="city_splitting"):
        ds.split_dataset("sample.jsonl", "jsonl", ratio=0.8, method="city_splitting", times=1)

    # Unsupported splitting method
    with pytest.raises(ValueError, match="not supported"):
        ds.split_dataset("sample.jsonl", "jsonl", ratio=0.8, method="unsupported_method", times=2)


def test_split_dataset_default_method():
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_file = os.path.join(tmp_dir, "data.txt")
        with open(data_file, "w", encoding="utf-8") as f:
            for i in range(10):
                f.write(f"item_{i}\n")

        ds = Dataset({})
        output_dir = os.path.join(tmp_dir, "split_output")

        res = ds.split_dataset(data_file, "txt", ratio=0.8, method="default", output_dir=output_dir, times=2)

        assert len(res) == 2
        for train_f, eval_f in res:
            assert os.path.exists(train_f)
            assert os.path.exists(eval_f)


def test_write_data_file_parent_directory_creation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        nested_file = os.path.join(tmp_dir, "subdir1", "subdir2", "data.txt")
        data = ["line1", "line2"]

        Dataset._write_data_file(data, nested_file, "txt")

        assert os.path.exists(nested_file)
        with open(nested_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f.readlines()]
        assert lines == data


def test_read_data_file_unsupported_format():
    with pytest.raises(ValueError, match="unsupported"):
        Dataset._read_data_file("dummy.xyz", "xyz")


def test_process_dataset_role_qualified_subdirectories():
    with tempfile.TemporaryDirectory() as tmp_dir:
        train_index = os.path.join(tmp_dir, "train_raw.txt")
        test_index = os.path.join(tmp_dir, "test_raw.txt")
        output_dir = os.path.join(tmp_dir, "staging")

        with open(train_index, "w", encoding="utf-8") as f:
            f.write("./img1.jpg 1\n")

        with open(test_index, "w", encoding="utf-8") as f:
            f.write("./img2.jpg 2\n")

        config = {
            "train_index": train_index,
            "test_index": test_index
        }
        ds = Dataset(config)
        ds.process_dataset(output_dir=output_dir)

        assert ds.train_url != ds.test_url
        assert os.path.dirname(ds.train_url) == os.path.join(output_dir, "train")
        assert os.path.dirname(ds.test_url) == os.path.join(output_dir, "test")
        assert os.path.exists(ds.train_url)
        assert os.path.exists(ds.test_url)
