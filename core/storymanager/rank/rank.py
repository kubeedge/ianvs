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

"""Rank"""

import copy
import os

import numpy as np
import pandas as pd

from core.common import utils
from core.storymanager.visualization import get_visualization_func


# pylint: disable=R0902
class Rank:
    """
    Rank:
    the output management and presentation of the test case,
    e.g., leaderboards

    """

    def __init__(self, config):
        self.sort_by: list = []
        self.visualization: dict = {
            "mode": "selected_only",
            "method": "print_table"
        }
        self.selected_dataitem: dict = {
            "paradigms": ["all"],
            "modules": ["all"],
            "hyperparameters": ["all"],
            "metrics": ["all"]
        }
        self.save_mode: str = "selected_and_all"

        self.all_df_header = None
        self.all_df = None
        self.all_rank_file = None
        self.selected_rank_file = None
        self._parse_config(config)

    def _parse_config(self, config):
        for attribute, value in config.items():
            if attribute in self.__dict__:
                self.__dict__[attribute] = value

        self._check_fields()

    def _check_fields(self):
        if not self.sort_by and not isinstance(self.sort_by, list):
            raise ValueError(f"rank's sort_by({self.sort_by}) must be provided and be list type.")

        if not self.visualization and not isinstance(self.visualization, dict):
            raise ValueError(f"rank's visualization({self.visualization}) "
                             f"must be provided and be dict type.")

        if not self.selected_dataitem and not isinstance(self.selected_dataitem, dict):
            raise ValueError(f"rank's selected_dataitem({self.selected_dataitem}) "
                             f"must be provided and be dict type.")

        if not self.selected_dataitem.get("paradigms"):
            raise ValueError("not found paradigms of selected_dataitem in rank.")

        if not self.selected_dataitem.get("modules"):
            raise ValueError("not found modules of selected_dataitem in rank.")

        if not self.selected_dataitem.get("metrics"):
            raise ValueError("not found metrics of selected_dataitem in rank.")

        if not self.save_mode and not isinstance(self.save_mode, list):
            raise ValueError(f"rank's save_mode({self.save_mode}) "
                             f"must be provided and be list type.")

    @classmethod
    def _get_all_metric_names(cls, test_results) -> list:
        metrics = set()
        # pylint: disable=C0103
        for _, v in test_results.items():
            metrics.update(v[0].keys())
        return list(metrics)

    @classmethod
    def _get_all_module_kinds(cls, test_cases) -> list:
        all_module_kinds = []
        for test_case in test_cases:
            modules = test_case.algorithm.modules
            for module_kind in modules.keys():
                if module_kind not in all_module_kinds:
                    all_module_kinds.append(module_kind)
        return all_module_kinds

    @classmethod
    def _get_algorithm_hyperparameters(cls, algorithm):
        hps = {}
        for module in algorithm.modules.values():
            hps.update(**module.hyperparameters)
        return hps

    def _get_all_hps_names(self, test_cases) -> list:
        all_hps_names = []
        for test_case in test_cases:
            algorithm = test_case.algorithm
            hps = self._get_algorithm_hyperparameters(algorithm)
            hps_names = hps.keys()

            for hps_name in hps_names:
                if hps_name not in all_hps_names:
                    all_hps_names.append(hps_name)
        return all_hps_names

    def _sort_all_df(self, all_df, all_metric_names):
        sort_metric_list = []
        is_ascend_list = []
        for ele in self.sort_by:
            metric_name = next(iter(ele))

            if metric_name not in all_metric_names:
                continue

            sort_metric_list.append(metric_name)
            is_ascend_list.append(ele.get(metric_name) == "ascend")

        return all_df.sort_values(by=sort_metric_list, ascending=is_ascend_list)

    def _get_all(self, test_cases, test_results) -> pd.DataFrame:
        all_df = pd.DataFrame(columns=self.all_df_header)
        for i, test_case in enumerate(test_cases):
            all_df.loc[i] = [np.NAN for i in range(len(self.all_df_header))]
            # fill name column of algorithm
            algorithm = test_case.algorithm
            all_df.loc[i][0] = algorithm.name
            # fill metric columns of algorithm
            for metric_name in test_results[test_case.id][0]:
                all_df.loc[i][metric_name] = test_results[test_case.id][0].get(metric_name)

            # file paradigm column of algorithm
            all_df.loc[i]["paradigm"] = algorithm.paradigm_kind

            # fill module columns of algorithm
            for module_kind, module in algorithm.modules.items():
                all_df.loc[i][module_kind] = module.name

            # fill hyperparameters columns of algorithm modules
            hps = self._get_algorithm_hyperparameters(algorithm)

            # pylint: disable=C0103
            for k, v in hps.items():
                all_df.loc[i][k] = v
            # fill time and output dir of testcase
            all_df.loc[i][-2:] = [test_results[test_case.id][1], test_case.output_dir]

        if utils.is_local_file(self.all_rank_file):
            old_df = pd.read_csv(self.all_rank_file, delim_whitespace=True, index_col=0)
            all_df = all_df.append(old_df)

        return self._sort_all_df(all_df, self._get_all_metric_names(test_results))

    def _save_all(self):
        all_df = copy.deepcopy(self.all_df)
        all_df.index = pd.np.arange(1, len(all_df) + 1)
        all_df.to_csv(self.all_rank_file, index_label="rank", encoding="utf-8", sep=" ")

    def _get_selected(self, test_cases, test_results) -> pd.DataFrame:
        module_kinds = self.selected_dataitem.get("modules")
        if module_kinds == ["all"]:
            module_kinds = self._get_all_module_kinds(test_cases)

        hps_names = self.selected_dataitem.get("hyperparameters")
        if hps_names == ["all"]:
            hps_names = self._get_all_hps_names(test_cases)

        metric_names = self.selected_dataitem.get("metrics")
        if metric_names == ["all"]:
            metric_names = self._get_all_metric_names(test_results)

        header = ["algorithm", *metric_names, "paradigm", *module_kinds, *hps_names, "time", "url"]

        all_df = copy.deepcopy(self.all_df)
        selected_df = pd.DataFrame(all_df, columns=header)
        selected_df = selected_df.drop_duplicates(header[:-2])

        paradigms = self.selected_dataitem.get("paradigms")
        if paradigms != ["all"]:
            selected_df = selected_df.loc[selected_df["paradigm"].isin(paradigms)]
        return selected_df

    def _save_selected(self, test_cases, test_results):
        selected_df = self._get_selected(test_cases, test_results)
        selected_df.index = pd.np.arange(1, len(selected_df) + 1)
        selected_df.to_csv(self.selected_rank_file, index_label="rank", encoding="utf-8", sep=" ")

    def _prepare(self, test_cases, test_results, output_dir):
        all_metric_names = self._get_all_metric_names(test_results)
        all_hps_names = self._get_all_hps_names(test_cases)
        all_module_kinds = self._get_all_module_kinds(test_cases)
        self.all_df_header = ["algorithm", *all_metric_names, "paradigm",
                              *all_module_kinds, *all_hps_names, "time", "url"]

        rank_output_dir = os.path.join(output_dir, "rank")
        if not utils.is_local_dir(rank_output_dir):
            os.makedirs(rank_output_dir)

        self.all_rank_file = os.path.join(rank_output_dir, "all_rank.csv")
        self.selected_rank_file = os.path.join(rank_output_dir, "selected_rank.csv")

        self.all_df = self._get_all(test_cases, test_results)

    def save(self, test_cases, test_results, output_dir):
        """
        save rank according to the save mode, include:
        e.g.: "selected_and_all", "selected_only"

        Parameters:
        ----------
        test_cases: list
        test_results: list
        output_dir: string

        """

        self._prepare(test_cases, test_results, output_dir)

        if self.save_mode == "selected_and_all":
            self._save_all()
            self._save_selected(test_cases, test_results)

        if self.save_mode == "selected_only":
            self._save_selected(test_cases, test_results)

    def plot(self):
        """
        plot rank according to the visual method, include
        e.g.: print_table

        """

        method = self.visualization.get("method")
        if self.visualization.get("mode") == "selected_only":
            try:
                func = get_visualization_func(method)
                func(self.selected_rank_file)
            except Exception as err:
                raise Exception(
                    f"process visualization(method={method}) of "
                    f"rank file({self.selected_rank_file}) failed, error: {err}.") from err
