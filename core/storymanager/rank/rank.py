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

import numpy as np
import pandas as pd

from core.storymanager.visualization import get_visualization_func


class Rank:
    def __init__(self):
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

    def check_fields(self):
        if not self.sort_by and not isinstance(self.sort_by, list):
            raise ValueError(f"the field of rank sort_by({self.sort_by}) is unvaild.")
        if not self.visualization and not isinstance(self.visualization, dict):
            raise ValueError(f"the field of rank visualization({self.visualization}) is unvaild.")
        if not self.selected_dataitem and not isinstance(self.selected_dataitem, dict):
            raise ValueError(f"the field of rank selected_dataitem({self.selected_dataitem}) is unvaild.")
        if not self.selected_dataitem.get("paradigms"):
            raise ValueError(f"not found paradigms of selected_dataitem in rank.")
        if not self.selected_dataitem.get("modules"):
            raise ValueError(f"not found modules of selected_dataitem in rank.")
        if not self.selected_dataitem.get("metrics"):
            raise ValueError(f"not found metrics of selected_dataitem in rank.")
        if not self.save_mode and not isinstance(self.save_mode, list):
            raise ValueError(f"the field of rank save_mode({self.save_mode}) is unvaild.")

    def _get_all_metric_names(self, test_results) -> list:
        metrics = set()
        for _, v in test_results.items():
            metrics.update(v[0].keys())
        return list(metrics)

    def _get_header(self, test_cases, test_results) -> list:
        all_metric_names = self._get_all_metric_names(test_results)
        all_module_kinds = self._get_all_module_kinds(test_cases)
        all_hps_names = self._get_all_hps_names(test_cases)
        all_df_header = ["algorithm", *all_metric_names, "paradigm", *all_module_kinds, *all_hps_names, "time", "url"]
        return all_df_header

    def _get_all_module_kinds(self, test_cases) -> list:
        all_module_kinds = []
        for test_case in test_cases:
            algorithm = test_case.algorithm
            for module in algorithm.modules:
                if module.kind not in all_module_kinds:
                    all_module_kinds.append(module.kind)
        return all_module_kinds

    def _get_all_hps_names(self, test_cases) -> list:
        all_hps_names = []
        for test_case in test_cases:
            algorithm = test_case.algorithm
            hps_names = algorithm.basemodel.hyperparameters.keys()
            for hps_name in hps_names:
                if hps_name not in all_hps_names:
                    all_hps_names.append(hps_name)
        return all_hps_names

    def _get_all(self, test_cases, test_results) -> pd.DataFrame:
        all_metric_names = self._get_all_metric_names(test_results)
        all_df_header = self._get_header(test_cases, test_results)
        all_df = pd.DataFrame(columns=all_df_header)
        for i, test_case in enumerate(test_cases):
            all_df.loc[i] = [np.NAN for i in range(len(all_df_header))]
            # fill name column of algorithm
            algorithm = test_case.algorithm
            all_df.loc[i][0] = algorithm.name
            # fill metric columns of algorithm
            test_result = test_results[test_case.id]
            for metric_name in test_result[0].keys():
                all_df.loc[i][metric_name] = test_result[0].get(metric_name)

            # file paradigm column of algorithm
            all_df.loc[i]["paradigm"] = algorithm.paradigm

            # fill module columns of algorithm
            for module in algorithm.modules:
                all_df.loc[i][module.kind] = module.name
            # fill hyperparameters columns of basemodel module of algorithm)
            hps = algorithm.basemodel.hyperparameters
            for k, v in hps.items():
                all_df.loc[i][k] = v
            # fill time and output dir of testcase
            all_df.loc[i][-2:] = [test_result[1], test_case.output_dir]

        if os.path.exists(self.all_rank_file):
            old_df = pd.read_csv(self.all_rank_file, delim_whitespace=True, index_col=0)
            all_df = all_df.append(old_df)

        for ele in self.sort_by:
            for k, v in ele.items():
                if k in all_metric_names:
                    all_df = all_df.sort_values(by=k, ascending=(v == "ascend"))
                # only one element
                break

        return all_df

    def _save_all(self, test_cases, test_results):
        dir = os.path.dirname(self.all_rank_file)
        if not os.path.exists(dir):
            os.makedirs(dir)
        all_df = self._get_all(test_cases, test_results)
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
        all_df = self._get_all(test_cases, test_results)
        selected_df = pd.DataFrame(all_df, columns=header)
        selected_df = selected_df.drop_duplicates(header[:-2])

        paradigms = self.selected_dataitem.get("paradigms")
        if paradigms != ["all"]:
            selected_df = selected_df.loc[selected_df["paradigm"].isin(paradigms)]
        return selected_df

    def _save_selected(self, test_cases, test_results):
        dir = os.path.dirname(self.selected_rank_file)
        if not os.path.exists(dir):
            os.makedirs(dir)

        selected_df = self._get_selected(test_cases, test_results)
        selected_df.index = pd.np.arange(1, len(selected_df) + 1)
        selected_df.to_csv(self.selected_rank_file, index_label="rank", encoding="utf-8", sep=" ")

    def _set_file(self, output_dir):
        self.all_rank_file = os.path.join(output_dir, "rank", "all_rank.csv")
        self.selected_rank_file = os.path.join(output_dir, "rank", "selected_rank.csv")

    def save(self, test_cases, test_results, output_dir):
        self._set_file(output_dir)
        if self.save_mode == "selected_and_all":
            self._save_all(test_cases, test_results)
            self._save_selected(test_cases, test_results)
        elif self.save_mode == "selected_only":
            self._save_selected(test_cases, test_results)

    def plot(self):
        method = self.visualization.get("method")
        if self.visualization.get("mode") == "selected_only":
            try:
                func = get_visualization_func(method)
                func(self.selected_rank_file)
            except Exception as err:
                raise Exception(
                    f"process visualization(method={method}) of rank file({self.selected_rank_file}) failed, error: {err}.")
