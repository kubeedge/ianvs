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

"""Algorithm"""

import copy

from core.common.constant import ParadigmKind
from core.testcasecontroller.algorithm.module import Module
from core.testcasecontroller.algorithm.paradigm import SingleTaskLearning, IncrementalLearning
from core.testcasecontroller.generation_assistant import get_full_combinations

# pylint: disable=too-few-public-methods
class Algorithm:
    """
    Algorithm:
    typical distributed-synergy AI algorithm paradigm.

    Parameters
    ----------
    name : string
        name of the algorithm paradigm
    config : dict
         config of the algorithm paradigm, includes: paradigm kind, modules, etc.
    """

    def __init__(self, name, config):
        self.name = name
        self.paradigm_kind: str = ""
        self.incremental_learning_data_setting: dict = {
            "train_ratio": 0.8,
            "splitting_method": "default"
        }
        self.initial_model_url: str = ""
        self.modules: list = []
        self.modules_list = None
        self._parse_config(config)

    def paradigm(self, workspace: str, **kwargs):
        """
        get test process of AI algorithm paradigm.

        Parameters:
        ----------
        workspace: string
            the output of test
        kwargs: dict
            config required for the test process of AI algorithm paradigm.

        Returns:
        -------
        the process of AI algorithm paradigm: instance

        """

        config = kwargs
        # pylint: disable=C0103
        for k, v in self.__dict__.items():
            config.update({k: v})

        if self.paradigm_kind == ParadigmKind.SINGLE_TASK_LEARNING.value:
            return SingleTaskLearning(workspace, **config)

        if self.paradigm_kind == ParadigmKind.INCREMENTAL_LEARNING.value:
            return IncrementalLearning(workspace, **config)

        return None

    def _check_fields(self):
        if not self.name and not isinstance(self.name, str):
            raise ValueError(f"algorithm name({self.name}) must be provided and be string type.")

        if not self.paradigm_kind and not isinstance(self.paradigm_kind, str):
            raise ValueError(
                f"algorithm paradigm({self.paradigm_kind}) must be provided and be string type.")

        paradigm_kinds = [e.value for e in ParadigmKind.__members__.values()]
        if self.paradigm_kind not in paradigm_kinds:
            raise ValueError(f"not support paradigm({self.paradigm_kind})."
                             f"the following paradigms can be selected: {paradigm_kinds}")

        if not isinstance(self.incremental_learning_data_setting, dict):
            raise ValueError(
                f"algorithm incremental_learning_data_setting"
                f"({self.incremental_learning_data_setting} must be dictionary type.")

        if not isinstance(self.initial_model_url, str):
            raise ValueError(
                f"algorithm initial_model_url({self.initial_model_url}) must be string type.")

    def _parse_config(self, config):
        config_dict = config[str.lower(Algorithm.__name__)]
        # pylint: disable=C0103
        for k, v in config_dict.items():
            if k == str.lower(Module.__name__ + "s"):
                self.modules_list = self._parse_modules_config(v)
            if k in self.__dict__:
                self.__dict__[k] = v
        self._check_fields()

    @classmethod
    def _parse_modules_config(cls, config):
        modules = []
        for module_config in config:
            module = Module(module_config)
            modules.append(module)

        modules_list = []
        for module in modules:
            hps_list = module.hyperparameters_list
            if not hps_list:
                modules_list.append((module.kind, None))
                continue

            module_list = []
            for hps in hps_list:
                new_module = copy.deepcopy(module)
                new_module.hyperparameters = hps
                module_list.append(new_module)

            modules_list.append((module.kind, module_list))

        module_combinations_list = get_full_combinations(modules_list)

        return module_combinations_list
