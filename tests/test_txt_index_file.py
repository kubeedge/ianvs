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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.testenvmanager.dataset import Dataset


def test_process_txt_index_file_path_label_disambiguation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        index_file = os.path.join(tmp_dir, "train_index.txt")
        lines = [
            "./images/img01.jpg 1\n",
            "./images/img02.jpg 0\n",
            "./images/img03.jpg ./annotations/ann03.xml\n"
        ]
        with open(index_file, "w", encoding="utf-8") as f:
            f.writelines(lines)

        processed_file = Dataset._process_txt_index_file(index_file)

        assert os.path.exists(processed_file)
        with open(processed_file, "r", encoding="utf-8") as f:
            processed_lines = [l.strip() for l in f.readlines()]

        expected_img01_path = os.path.abspath(os.path.join(tmp_dir, "./images/img01.jpg"))
        expected_img02_path = os.path.abspath(os.path.join(tmp_dir, "./images/img02.jpg"))
        expected_img03_path = os.path.abspath(os.path.join(tmp_dir, "./images/img03.jpg"))
        expected_ann03_path = os.path.abspath(os.path.join(tmp_dir, "./annotations/ann03.xml"))

        # Verify image paths are absolute and non-path labels ("1", "0") are preserved
        assert processed_lines[0] == f"{expected_img01_path} 1"
        assert processed_lines[1] == f"{expected_img02_path} 0"
        assert processed_lines[2] == f"{expected_img03_path} {expected_ann03_path}"


def test_process_txt_index_file_cifar100_tab_format():
    with tempfile.TemporaryDirectory() as tmp_dir:
        index_file = os.path.join(tmp_dir, "cifar100_train.txt")
        lines = [
            "cifar100_train_index_0.npy\t0\n",
            "cifar100_train_index_1.npy\t1\n"
        ]
        with open(index_file, "w", encoding="utf-8") as f:
            f.writelines(lines)

        processed_file = Dataset._process_txt_index_file(index_file)
        with open(processed_file, "r", encoding="utf-8") as f:
            processed_lines = f.readlines()

        expected_0 = os.path.abspath(os.path.join(tmp_dir, "cifar100_train_index_0.npy"))
        expected_1 = os.path.abspath(os.path.join(tmp_dir, "cifar100_train_index_1.npy"))

        assert processed_lines[0] == f"{expected_0}\t0\n"
        assert processed_lines[1] == f"{expected_1}\t1\n"


def test_process_txt_index_file_absolute_first_relative_second():
    with tempfile.TemporaryDirectory() as tmp_dir:
        index_file = os.path.join(tmp_dir, "dataset_index.txt")
        abs_img = os.path.abspath(os.path.join(tmp_dir, "img.jpg"))
        rel_ann = "./annotations/ann.xml"
        lines = [
            f"{abs_img} {rel_ann}\n"
        ]
        with open(index_file, "w", encoding="utf-8") as f:
            f.writelines(lines)

        processed_file = Dataset._process_txt_index_file(index_file)
        with open(processed_file, "r", encoding="utf-8") as f:
            processed_lines = f.readlines()

        expected_ann = os.path.abspath(os.path.join(tmp_dir, rel_ann))
        assert processed_lines[0] == f"{abs_img} {expected_ann}\n"


def test_process_txt_index_file_directory_name_collision_with_label():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create an unrelated local directory named "cat"
        os.makedirs(os.path.join(tmp_dir, "cat"), exist_ok=True)

        index_file = os.path.join(tmp_dir, "train_index.txt")
        lines = [
            "./images/img.jpg cat\n"
        ]
        with open(index_file, "w", encoding="utf-8") as f:
            f.writelines(lines)

        processed_file = Dataset._process_txt_index_file(index_file)
        with open(processed_file, "r", encoding="utf-8") as f:
            processed_lines = f.readlines()

        expected_img = os.path.abspath(os.path.join(tmp_dir, "./images/img.jpg"))
        # Label "cat" must remain unchanged as label, NOT converted to tmp_dir/cat
        assert processed_lines[0] == f"{expected_img} cat\n"
