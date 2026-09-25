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

"""Dataset"""

import os
import tempfile

import pandas as pd
from sedna.datasources import (
    CSVDataParse,
    TxtDataParse,
    JSONDataParse,
    JsonlDataParse,
    JSONMetaDataParse
)

from core.common import utils
from core.common.constant import DatasetFormat

# pylint: disable=too-many-instance-attributes
class Dataset:
    """
    Data:
    provide the configuration and handle functions of dataset.

    Parameters
    ----------
    config : dict
         config of dataset, include: train url, test url and label, etc.
    """

    def __init__(self, config):
        self.train_url: str = ""
        self.test_url: str = ""
        self.train_index: str = ""
        self.test_index: str = ""
        self.train_data: str = ""
        self.test_data: str = ""
        self.train_data_info: str = ""
        self.test_data_info: str = ""
        self.label: str = ""
        self._parse_config(config)

    def _check_fields(self):
        if self.train_index:
            self._check_dataset_url(self.train_index)
        if self.test_index:
            self._check_dataset_url(self.test_index)
        if self.train_data:
            self._check_dataset_url(self.train_data)
        if self.test_data:
            self._check_dataset_url(self.test_data)
        if self.train_data_info:
            self._check_dataset_url(self.train_data_info)
        if self.test_data_info:
            self._check_dataset_url(self.test_data_info)

    def _parse_config(self, config):
        for attr, value in config.items():
            if attr in self.__dict__:
                self.__dict__[attr] = value

        self._check_fields()

    @classmethod
    def _check_dataset_url(cls, url):
        if not utils.is_local_file(url) and not os.path.isabs(url):
            raise ValueError(
                f"dataset file({url}) is not a local file and not a absolute path."
            )

        file_format = utils.get_file_format(url)
        if file_format not in [v.value for v in DatasetFormat.__members__.values()]:
            raise ValueError(
                f"dataset file({url})'s format({file_format}) is not supported."
            )

    @classmethod
    def _process_txt_index_file(cls, file_url, output_dir=None):
        """
        convert the index info of data from relative path to absolute path in txt index file
        """
        flag = False
        new_file = file_url
        with open(file_url, "r", encoding="utf-8") as file:
            lines = file.readlines()
            for line in lines:
                if not os.path.isabs(line.split(" ")[0]):
                    flag = True
                    break
        if flag:
            root = os.path.dirname(file_url)
            if not output_dir:
                tmp_dir = tempfile.mkdtemp()
            else:
                tmp_dir = output_dir
                os.makedirs(tmp_dir, exist_ok=True)

            tmp_file = os.path.join(tmp_dir, "index.txt")
            with open(tmp_file, "w", encoding="utf-8") as file:
                for line in lines:
                    # copy all the files in the line
                    line = line.strip()
                    words = line.split(" ")
                    length = len(words)
                    words[-1] = words[-1] + "\n"
                    for i in range(length):
                        file.writelines(
                            f"{os.path.abspath(os.path.join(root, words[i]))}"
                        )
                        if i < length - 1:
                            file.writelines(" ")

            new_file = tmp_file

        return new_file

    def _process_index_file(self, file_url, output_dir=None):
        file_format = utils.get_file_format(file_url)
        if file_format == DatasetFormat.TXT.value:
            return self._process_txt_index_file(file_url, output_dir=output_dir)
        if file_format == DatasetFormat.JSON.value:
            return file_url

        return None

    def _process_data_file(self, file_url):
        file_format = utils.get_file_format(file_url)
        if file_format == DatasetFormat.JSONL.value:
            return file_url

        return None

    def _process_data_info_file(self, file_url):
        file_format = utils.get_file_format(file_url)
        if file_format == DatasetFormat.JSONFORLLM.value:
            return file_url
        raise ValueError(
            f"The Data Info File must be named as `data_info.json`, "
            f"but the current file is {file_url}."
        )

    def process_dataset(self, output_dir=None):
        """
        process dataset:
        process train dataset and test dataset for testcase;
        e.g.: convert the index info of data from relative path to absolute path
              in the index file(e.g.: txt index file).

        """
        if self.train_index:
            train_output_dir = os.path.join(output_dir, "train") if output_dir else None
            self.train_url = self._process_index_file(self.train_index, output_dir=train_output_dir)
        elif self.train_data:
            self.train_url = self._process_data_file(self.train_data)
        elif self.train_data_info:
            self.train_url = self._process_data_info_file(self.train_data_info)
            # raise NotImplementedError('to be done')
        else:
            raise NotImplementedError('not one of train_index/train_data/train_data_info')

        if self.test_index:
            test_output_dir = os.path.join(output_dir, "test") if output_dir else None
            self.test_url = self._process_index_file(self.test_index, output_dir=test_output_dir)
        elif self.test_data:
            self.test_url = self._process_data_file(self.test_data)
        elif self.test_data_info:
            self.test_url = self._process_data_info_file(self.test_data_info)
            # raise NotImplementedError('to be done')
        else:
            raise NotImplementedError('not one of test_index/test_data/test_data_info')

    # pylint: disable=too-many-arguments
    def split_dataset(
        self,
        dataset_url,
        dataset_format,
        ratio,
        method="default",
        dataset_types=None,
        output_dir=None,
        times=1,
    ):
        """
        split dataset:
            step1: divide all data N(N = times) times to generate N pieces of data.
            step2: divide every pieces of data 1 time using the special method.

        Parameters:
        -----------
        dataset_url: str
            the address url of dataset.
        dataset_format: str
            the format of dataset, e.g.: txt, csv and jsonl.
        ratio: float
            the float of splitting dataset
        method: string
            the method of splitting dataset.
            default value is "default": divide the data equally and proportionally.
        dataset_types: tuple
            divide every pieces of data 1 time to generate 2 small pieces of  data
            for special types of tasks.
            e.g.: ("train", "eval")
        output_dir: str
            the output dir of splitting dataset.
        times: int
            the times of dividing all data in step1.

        Returns
        -------
        list
            the result of splitting dataset.
            e.g.: [("/dataset/train.txt", "/dataset/eval.txt")]

        """

        if not isinstance(times, int) or times < 1:
            raise ValueError(f"dataset splitting parameter 'times' ({times}) must be an integer >= 1.")

        if not isinstance(ratio, (int, float)) or not (0.0 < ratio < 1.0):
            raise ValueError(f"dataset splitting parameter 'ratio' ({ratio}) must be a float strictly between 0.0 and 1.0.")

        if method == "city_splitting" and times < 2:
            raise ValueError(f"dataset splitting method 'city_splitting' requires times >= 2, but got {times}.")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if method == "default":
            return self._splitting_more_times(
                dataset_url,
                dataset_format,
                ratio,
                data_types=dataset_types,
                output_dir=output_dir,
                times=times,
            )
        # add new splitting method for semantic segmantation
        if method == "city_splitting":
            return self._city_splitting(
                dataset_url,
                dataset_format,
                ratio,
                data_types=dataset_types,
                output_dir=output_dir,
                times=times,
            )
        if method == "fwt_splitting":
            return self._fwt_splitting(
                dataset_url,
                dataset_format,
                ratio,
                data_types=dataset_types,
                output_dir=output_dir,
                times=times,
            )

        if method == "hard-example_splitting":
            return self._hard_example_splitting(
                dataset_url,
                dataset_format,
                ratio,
                data_types=dataset_types,
                output_dir=output_dir,
                times=times,
            )

        raise ValueError(
            f"dataset splitting method({method}) is not supported,"
            f"currently supported methods: ['default', 'city_splitting', 'fwt_splitting', 'hard-example_splitting']."
        )

    @classmethod
    def _get_file_url(cls, output_dir, dataset_type, dataset_id, file_format):
        return os.path.join(output_dir, f"{dataset_type}-{dataset_id}.{file_format}")

    @classmethod
    def _write_data_file(cls, data, data_file, data_format):
        dir_name = os.path.dirname(data_file)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        if data_format in (DatasetFormat.TXT.value, DatasetFormat.JSONL.value):
            with open(data_file, "w", encoding="utf-8") as file:
                for line in data:
                    file.writelines(str(line) + "\n")
        elif data_format == DatasetFormat.CSV.value:
            data.to_csv(data_file, index=None)
        else:
            raise ValueError(f"unsupported data format ({data_format}) for dataset writing.")

    @classmethod
    def _read_data_file(cls, data_file, data_format):
        data = None

        if data_format in (DatasetFormat.TXT.value, DatasetFormat.JSONL.value):
            with open(data_file, "r", encoding="utf-8") as file:
                data = [line.strip() for line in file.readlines()]

        elif data_format == DatasetFormat.CSV.value:
            data = pd.read_csv(data_file)
        else:
            raise ValueError(f"unsupported dataset format ({data_format}) for dataset reading.")

        if data is None:
            raise ValueError(f"failed to read dataset from file ({data_file}) with format ({data_format}).")

        return data

    def _get_dataset_file(self, data, output_dir, dataset_type, index, dataset_format):
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        data_file = self._get_file_url(output_dir, dataset_type, index, dataset_format)

        self._write_data_file(data, data_file, dataset_format)

        return data_file

    def _splitting_more_times(
        self, data_file, data_format, ratio, data_types=None, output_dir=None, times=1
    ):
        if not data_types:
            data_types = ("train", "eval")

        if not output_dir:
            output_dir = tempfile.mkdtemp()
        else:
            os.makedirs(output_dir, exist_ok=True)

        all_data = self._read_data_file(data_file, data_format)

        data_files = []

        all_num = len(all_data)
        step = int(all_num / times)
        index = 1
        while index <= times:
            if index == times:
                new_dataset = all_data[step * (index - 1) :]
            else:
                new_dataset = all_data[step * (index - 1) : step * index]

            new_num = len(new_dataset)

            data_files.append(
                (
                    self._get_dataset_file(
                        new_dataset[: int(new_num * ratio)],
                        output_dir,
                        data_types[0],
                        index,
                        data_format,
                    ),
                    self._get_dataset_file(
                        new_dataset[int(new_num * ratio) :],
                        output_dir,
                        data_types[1],
                        index,
                        data_format,
                    ),
                )
            )

            index += 1

        return data_files

    def _fwt_splitting(
        self, data_file, data_format, ratio, data_types=None, output_dir=None, times=1
    ):
        if not data_types:
            data_types = ("train", "eval")

        if not output_dir:
            output_dir = tempfile.mkdtemp()
        else:
            os.makedirs(output_dir, exist_ok=True)

        all_data = self._read_data_file(data_file, data_format)

        data_files = []

        all_num = len(all_data)
        step = int(all_num / times)
        data_files.append(
            (
                self._get_dataset_file(
                    all_data[:1], output_dir, data_types[0], 0, data_format
                ),
                self._get_dataset_file(
                    all_data[:1], output_dir, data_types[1], 0, data_format
                ),
            )
        )
        index = 1
        while index <= times:
            if index == times:
                new_dataset = all_data[step * (index - 1) :]
            else:
                new_dataset = all_data[step * (index - 1) : step * index]

            new_num = len(new_dataset)

            data_files.append(
                (
                    self._get_dataset_file(
                        new_dataset[: int(new_num * ratio)],
                        output_dir,
                        data_types[0],
                        index,
                        data_format,
                    ),
                    self._get_dataset_file(
                        new_dataset[int(new_num * ratio) :],
                        output_dir,
                        data_types[1],
                        index,
                        data_format,
                    ),
                )
            )

            index += 1

        return data_files

    # add new splitting method for semantic segmentation
    def _city_splitting(
        self, data_file, data_format, ratio, data_types=None, output_dir=None, times=1
    ):
        if not data_types:
            data_types = ("train", "eval")

        if not output_dir:
            output_dir = tempfile.mkdtemp()
        else:
            os.makedirs(output_dir, exist_ok=True)

        all_data = self._read_data_file(data_file, data_format)

        data_files = []

        index0 = 0
        for i, data in enumerate(all_data):
            if "synthia_sim" in data:
                continue
            index0 = i
            break

        new_dataset = all_data[:index0]
        data_files.append(
            (
                self._get_dataset_file(
                    new_dataset[: int(len(new_dataset) * ratio)],
                    output_dir,
                    data_types[0],
                    1,
                    data_format,
                ),
                self._get_dataset_file(
                    new_dataset[int(len(new_dataset) * ratio) :],
                    output_dir,
                    data_types[1],
                    1,
                    data_format,
                ),
            )
        )
        times = times - 1
        step = int((len(all_data) - index0) / times)
        index = 1
        while index <= times:
            if index == times:
                new_dataset = all_data[index0 + step * (index - 1) :]
            else:
                new_dataset = all_data[
                    index0 + step * (index - 1) : index0 + step * index
                ]

            data_files.append(
                (
                    self._get_dataset_file(
                        new_dataset[: int(len(new_dataset) * ratio)],
                        output_dir,
                        data_types[0],
                        index + 1,
                        data_format,
                    ),
                    self._get_dataset_file(
                        new_dataset[int(len(new_dataset) * ratio) :],
                        output_dir,
                        data_types[1],
                        index + 1,
                        data_format,
                    ),
                )
            )

            index += 1

        return data_files

    def _hard_example_splitting(
        self, data_file, data_format, ratio, data_types=None, output_dir=None, times=1
    ):
        if not data_types:
            data_types = ("train", "eval")

        if not output_dir:
            output_dir = tempfile.mkdtemp()
        else:
            os.makedirs(output_dir, exist_ok=True)

        all_data = self._read_data_file(data_file, data_format)

        data_files = []

        all_num = len(all_data)
        step = int(all_num / (times * 2))
        data_files.append(
            (
                self._get_dataset_file(
                    all_data[: int((all_num * ratio) / 2)],
                    output_dir,
                    data_types[0],
                    0,
                    data_format,
                ),
                self._get_dataset_file(
                    all_data[int((all_num * ratio) / 2) : int(all_num / 2)],
                    output_dir,
                    data_types[1],
                    0,
                    data_format,
                ),
            )
        )
        index = 1
        while index <= times:
            if index == times:
                new_dataset = all_data[int(all_num / 2) + step * (index - 1) :]
            else:
                new_dataset = all_data[
                    int(all_num / 2)
                    + step * (index - 1) : int(all_num / 2)
                    + step * index
                ]

            new_num = len(new_dataset)

            data_files.append(
                (
                    self._get_dataset_file(
                        new_dataset[: int(new_num * ratio)],
                        output_dir,
                        data_types[0],
                        index,
                        data_format,
                    ),
                    self._get_dataset_file(
                        new_dataset[int(new_num * ratio) :],
                        output_dir,
                        data_types[1],
                        index,
                        data_format,
                    ),
                )
            )

            index += 1

        return data_files

    @classmethod
    def load_data(cls, file: str, data_type: str, label=None,
                  use_raw=False, feature_process=None, **kwargs):
        """
        load data

        Parameters
        ---------
        file: str
            the address url of data file.
        data_type: str
            the type of data for special type task.
        label: str
            specify label of data.
        use_raw: bool
            if true, use all of raw data.
        feature_process: function
            feature processing on all of raw data.

        Returns
        -------
        instance
            e.g.: TxtDataParse, CSVDataParse.

        """
        data_format = utils.get_file_format(file)

        data = None
        if data_format == DatasetFormat.CSV.value:
            data = CSVDataParse(data_type=data_type, func=feature_process)
            data.parse(file, label=label)

        if data_format == DatasetFormat.TXT.value:
            data = TxtDataParse(data_type=data_type, func=feature_process)
            data.parse(file, use_raw=use_raw)

        if data_format == DatasetFormat.JSON.value:
            data = JSONDataParse(data_type=data_type, func=feature_process)
            data.parse(file)

        if data_format == DatasetFormat.JSONL.value:
            data = JsonlDataParse(data_type=data_type, func=feature_process)
            data.parse(file)

        if data_format == DatasetFormat.JSONFORLLM.value:
            data = JSONMetaDataParse(data_type=data_type, func=feature_process)
            data.parse(file, **kwargs)

        return data
