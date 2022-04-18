import os
import tempfile

import pandas as pd

from core.common import utils
from core.common.constant import DatasetFormat


class Dataset:
    def __init__(self):
        self.url: str = ""
        self.label: str = ""
        self.train_ratio: float = 0.8
        self.splitting_method: str = "default"

    def check_fields(self):
        if not self.url:
            raise ValueError(f"not found dataset url({self.url}).")

    def process_dataset(self, output_dir):
        if not utils.is_local_file(self.url):
            raise ValueError(f"dataset file({self.url}) is not the local file.")

        file_format = utils.get_file_format(self.url)
        if file_format not in [v.value for v in DatasetFormat.__members__.values()]:
            raise ValueError(f"dataset file({self.url})'s format({file_format}) is not supported.")

        self.format = file_format

        self.output_dir = os.path.join(output_dir, "dataset")
        all_output_dir = os.path.join(self.output_dir, "all")
        if not os.path.exists(all_output_dir):
            os.makedirs(all_output_dir)
        try:
            if self.splitting_method == "default":
                self._default_splitting_method(self.url, file_format, self.train_ratio, all_output_dir)
            else:
                raise ValueError(f"dataset splitting method({self.splitting_method}) is unvaild.")
        except Exception as err:
            raise Exception(f"split dataset failed, error:{err}")

    def _default_splitting_method(self, dataset_url, dataset_format, train_ratio, output_dir):
        dataset_files = self.splitting_more_times(dataset_url, dataset_format, train_ratio, output_dir)
        if len(dataset_files) == 1:
            self.train_dataset, self.eval_dataset = dataset_files[0]

    def splitting_more_times(self, dataset_url, dataset_format, ratio, output_dir=None, times=1):
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
