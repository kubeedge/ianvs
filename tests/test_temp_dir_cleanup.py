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

"""Unit tests ensuring paradigm/dataset staging files stay inside the workspace
and no orphaned directories are left in the OS temp root."""

import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from core.testcasecontroller.algorithm.paradigm.incremental_learning.incremental_learning import (
    IncrementalLearning,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class MockDataLabels:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class MockDataset:
    def __init__(self, data_labels):
        self._data_labels = data_labels

    def load_data(self, file, data_type):
        # pylint: disable=unused-argument
        return self._data_labels


def test_get_train_dataset_stays_in_workspace_and_leaves_no_orphan():
    with tempfile.TemporaryDirectory() as workspace:
        label_file = os.path.join(workspace, "hard_examples.txt")
        with open(label_file, "w", encoding="utf-8") as file:
            file.write("old.jpg cat\n")

        data_labels = MockDataLabels(
            x=np.array(["old.jpg"]), y=np.array(["cat"])
        )

        incremental_learning = IncrementalLearning.__new__(IncrementalLearning)
        incremental_learning.workspace = workspace
        incremental_learning.dataset = MockDataset(data_labels)

        before = set(os.listdir(tempfile.gettempdir()))

        train_dataset_file = incremental_learning._get_train_dataset(
            [("old.jpg", "new.jpg")], label_file, rounds=1
        )

        after = set(os.listdir(tempfile.gettempdir()))

        assert train_dataset_file.startswith(workspace)
        assert os.path.exists(train_dataset_file)
        # no new entries were left behind in the OS temp root
        assert after == before


def test_process_txt_index_file_cleans_up_temp_dir_on_exit():
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_file = os.path.join(tmp_dir, "sample.jpg")
        with open(data_file, "w", encoding="utf-8") as file:
            file.write("data")

        index_file = os.path.join(tmp_dir, "index.txt")
        with open(index_file, "w", encoding="utf-8") as file:
            # relative path triggers the absolute-path rewrite code path
            file.write("sample.jpg label\n")

        script = (
            "from core.testenvmanager.dataset.dataset import Dataset\n"
            f"print(Dataset._process_txt_index_file(r'{index_file}'))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            check=True,
        )

        produced_file = result.stdout.strip().splitlines()[-1]
        produced_dir = os.path.dirname(produced_file)

        assert produced_dir != os.path.dirname(index_file)
        # the child process' atexit cleanup must have removed the staging dir
        assert not os.path.exists(produced_dir)


def test_process_txt_index_file_with_explicit_output_dir():
    with tempfile.TemporaryDirectory() as workspace:
        data_file = os.path.join(workspace, "sample.jpg")
        with open(data_file, "w", encoding="utf-8") as file:
            file.write("data")

        index_file = os.path.join(workspace, "index.txt")
        with open(index_file, "w", encoding="utf-8") as file:
            file.write("sample.jpg label\n")

        staging_dir = os.path.join(workspace, "staging_dir")
        from core.testenvmanager.dataset.dataset import Dataset
        res_file = Dataset._process_txt_index_file(index_file, output_dir=staging_dir)

        assert res_file.startswith(staging_dir)
        assert os.path.exists(res_file)
