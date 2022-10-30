# Copyright 2021 The KubeEdge Authors.
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

"""
Remodeling tasks based on their relationships

Parameters
----------
mappings ：all assigned tasks get from the `task_mining`
samples : input samples

Returns
-------
models : List of groups which including at least 1 task.
"""

from typing import List
import numpy as np

from sedna.datasources import BaseDataSource
from sedna.common.class_factory import ClassFactory, ClassType

__all__ = ('TaskRemodeling',)
# class Model:
#     def __init__(self, index: int, entry, model, result):
#         self.index = index  # integer
#         self.entry = entry
#         self.model = model
#         self.result = result
#         self.meta_attr = None  # assign on running


@ClassFactory.register(ClassType.STP)
class TaskRemodeling:
    """
    Assume that each task is independent of each other
    """

    def __init__(self, **kwargs):
        self.models = []
        selected_model_path = "/mnt/disk/shifan/ianvs/yolo_model/"
        weight_list = ['all.pt', 'bdd.pt', 'traffic_0.pt', 'bdd_street.pt', 'bdd_clear.pt', 'bdd_daytime.pt',
                'bdd_highway.pt', 'traffic_2.pt', 'bdd_overcast.pt', 'bdd_residential.pt', 'traffic_1.pt', 
                'bdd_snowy.pt', 'bdd_rainy.pt', 'bdd_night.pt', 'soda.pt', 'bdd_cloudy.pt', 'bdd_cloudy_night.pt',
                'bdd_highway_residential.pt', 'bdd_snowy_rainy.pt', 'soda_t1.pt']
        for i in range(len(weight_list)):
            self.models.append([i, weight_list[i][:-3], selected_model_path + weight_list[i]])


    def __call__(self, samples: BaseDataSource, mappings: List):
        """
        Grouping based on assigned tasks
        """
        mappings = np.array(mappings)
        data, models = samples.x[0], []
        for m in mappings:
            try:
                model = self.models[m]
            except Exception as err:
                print(f"self.models[{m}] not exists. {err}")
                model = self.models[0]
            models.append(model)
        return data, models
