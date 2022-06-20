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

import os
import tempfile

import pandas as pd

from core.common import utils
from core.common.constant import DatasetFormat


class Dataset:
    def __init__(self):
        self.train_url: str = ""
        self.test_url: str = ""
        self.label: str = ""

    def check_fields(self):
        self._check_dataset_url(self.train_url)
        self._check_dataset_url(self.test_url)

    def _check_dataset_url(self, url):
        if not utils.is_local_file(url) and not os.path.isabs(url):
            raise ValueError(f"dataset file({url}) is not a local file and not a absolute path.")

        file_format = utils.get_file_format(url)
        if file_format not in [v.value for v in DatasetFormat.__members__.values()]:
            raise ValueError(f"dataset file({url})'s format({file_format}) is not supported.")

    def _process_txt_index_file(self, file_url):
        flag = False
        new_file = file_url
        with open(file_url, "r") as f:
            lines = f.readlines()
            for line in lines:
                if not os.path.isabs(line.split(" ")[0]):
                    flag = True
                    break
        if flag:
            root = os.path.dirname(file_url)
            tmp_file = os.path.join(tempfile.mkdtemp(), "index.txt")
            with open(tmp_file, "w") as f:
                for line in lines:
                    front, back = line.split(" ")
                    f.writelines(
                        f"{os.path.abspath(os.path.join(root, front))} {os.path.abspath(os.path.join(root, back))}")

            new_file = tmp_file

        return new_file

    def _process_index_file(self, file_url):
        file_format = utils.get_file_format(file_url)
        if file_format == DatasetFormat.TXT.value:
            return self._process_txt_index_file(file_url)

    def process_dataset(self):
        self.train_url = self._process_index_file(self.train_url)
        self.test_url = self._process_index_file(self.test_url)

    def splitting_dataset(self, dataset_url, dataset_format, ratio, method="default", output_dir=None, times=1):
        try:
            if method == "default":
                return self._splitting_more_times(dataset_url,
                                                  dataset_format,
                                                  ratio,
                                                  output_dir=output_dir,
                                                  times=times)
            else:
                raise ValueError(f"dataset splitting method({method}) is unvaild.")
        except Exception as err:
            raise Exception(f"split dataset failed, error:{err}")

    def _splitting_more_times(self, dataset_url, dataset_format, ratio, output_dir=None, times=1):
        if not output_dir:
            output_dir = tempfile.mkdtemp()

        def get_file_url(type, id, format):
            return os.path.join(output_dir, f"dataset-{type}-{id}.{format}")

        dataset_files = []
        if dataset_format == DatasetFormat.CSV.value:
            df = pd.read_csv(dataset_url)
            all_num = len(df)
            step = int(all_num / times)
            index = 1
            while index <= times:
                if index == times:
                    new_df = df[step * (index - 1):]
                else:
                    new_df = df[step * (index - 1):step * index]

                new_num = len(new_df)
                train_data = new_df[:int(new_num * ratio)]
                train_data_file = get_file_url("train", index, dataset_format)
                train_data.to_csv(train_data_file, index=None)

                eval_data = new_df[int(new_num * ratio):]
                eval_data_file = get_file_url("eval", index, dataset_format)
                eval_data.to_csv(eval_data_file, index=None)
                dataset_files.append((train_data_file, eval_data_file))

                index += 1
        elif dataset_format == DatasetFormat.TXT.value:
            with open(dataset_url, "r") as f:
                dataset = [line.strip() for line in f.readlines()]
                dataset.append(f.readline())

            all_num = len(dataset)
            step = int(all_num / times)
            index = 1

            while index <= times:
                if index == times:
                    new_dataset = dataset[step * (index - 1):]
                else:
                    new_dataset = dataset[step * (index - 1):step * index]

                new_num = len(new_dataset)
                train_data = new_dataset[:int(new_num * ratio)]
                eval_data = new_dataset[int(new_num * ratio):]

                def write_to_file(data, file):
                    with open(file, "w") as f:
                        for line in data:
                            f.writelines(line + "\n")

                train_data_file = get_file_url("train", index, dataset_format)
                write_to_file(train_data, train_data_file)
                eval_data_file = get_file_url("eval", index, dataset_format)
                write_to_file(eval_data, eval_data_file)
                dataset_files.append((train_data_file, eval_data_file))

                index += 1
        return dataset_files
