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

"""Pins S1: how Dataset parses its config (core/testenvmanager/dataset/dataset.py).

Preservation tests pass on the unmodified code and must still pass after S1.
Change tests fail on the unmodified code and pass after S1.
"""

import logging

import pytest

from core.common.log import LOGGER
from core.testenvmanager.dataset.dataset import Dataset


@pytest.fixture(name="index_files")
def fixture_index_files(tmp_path):
    """Two txt index files whose entries are absolute paths."""
    sample = tmp_path / "sample.png"
    sample.write_text("x", encoding="utf-8")
    files = {}
    for name in ("train", "test"):
        index = tmp_path / f"{name}-index.txt"
        index.write_text(f"{sample} {sample}\n", encoding="utf-8")
        files[name] = str(index)
    return files


@pytest.fixture(name="ianvs_logs")
def fixture_ianvs_logs(caplog):
    """The ianvs logger does not propagate; let caplog see it for one test."""
    LOGGER.propagate = True
    caplog.set_level(logging.WARNING, logger=LOGGER.name)
    yield caplog
    LOGGER.propagate = False


# Preservation tests


def test_modern_keys_are_unchanged(index_files):
    dataset = Dataset({"train_index": index_files["train"],
                       "test_index": index_files["test"]})

    assert dataset.train_index == index_files["train"]
    assert dataset.test_index == index_files["test"]
    assert dataset.train_url == ""
    assert dataset.test_url == ""

    dataset.process_dataset()
    assert dataset.train_url == index_files["train"]
    assert dataset.test_url == index_files["test"]


def test_modern_key_wins_over_legacy_key(index_files, tmp_path):
    other = tmp_path / "other-index.txt"
    other.write_text("", encoding="utf-8")
    dataset = Dataset({"train_index": index_files["train"],
                       "train_url": str(other),
                       "test_index": index_files["test"]})

    dataset.process_dataset()
    assert dataset.train_index == index_files["train"]
    assert dataset.train_url == index_files["train"]


def test_modern_keys_log_no_warning(index_files, ianvs_logs):
    Dataset({"train_index": index_files["train"],
             "test_index": index_files["test"]})

    assert not ianvs_logs.records


def test_missing_train_input_still_raises(index_files):
    dataset = Dataset({"test_index": index_files["test"]})

    with pytest.raises(NotImplementedError):
        dataset.process_dataset()


# Change tests


def test_legacy_keys_map_to_index(index_files):
    dataset = Dataset({"train_url": index_files["train"],
                       "test_url": index_files["test"]})

    assert dataset.train_index == index_files["train"]
    assert dataset.test_index == index_files["test"]

    dataset.process_dataset()
    assert dataset.train_url == index_files["train"]
    assert dataset.test_url == index_files["test"]


def test_legacy_keys_log_deprecation_warning(index_files, ianvs_logs):
    Dataset({"train_url": index_files["train"],
             "test_url": index_files["test"]})

    messages = [record.getMessage() for record in ianvs_logs.records]
    assert any("train_url" in m and "train_index" in m for m in messages)
    assert any("test_url" in m and "test_index" in m for m in messages)
