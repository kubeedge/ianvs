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

"""Paradigm Base"""

import os

from sedna.core.incremental_learning import IncrementalLearning

from core.common.constant import ModuleKind, ParadigmKind


class ParadigmBase:
    """
    Paradigm Base

    Parameters
    ---------
    workspace: string
        the output required for test process of AI algorithm paradigm.
    kwargs: dict
        config required for the test process of AI algorithm paradigm,
        e.g.: algorithm modules, dataset, etc.

    """

    def __init__(self, workspace, **kwargs):
        self.modules = kwargs.get("modules")
        self.dataset = kwargs.get("dataset")
        self.workspace = workspace
        self.system_metric_info = {}
        self.modules_funcs = self._get_module_funcs()
        os.environ["LOCAL_TEST"] = "TRUE"

    def dataset_output_dir(self):
        """
        get output dir of dataset in test process

        Returns
        ------
        str

        """
        output_dir = os.path.join(self.workspace, "dataset")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return output_dir

    def _get_module_funcs(self):
        module_funcs = {}
        for module_kind, module in self.modules.items():
            func = module.get_module_func(module_kind)
            if callable(func):
                module_funcs.update({module_kind: func})
        return module_funcs

    def build_paradigm_job(self, paradigm_kind):
        """
        build paradigm job instance according to paradigm kind.
        this job instance provides the test flow of some algorithm modules.

        Parameters
        ---------
        paradigm_kind: str

        Returns
        -------
        instance

        """
        if paradigm_kind == ParadigmKind.SINGLE_TASK_LEARNING.value:
            return self.modules_funcs.get(ModuleKind.BASEMODEL.value)()

        if paradigm_kind == ParadigmKind.INCREMENTAL_LEARNING.value:
            return IncrementalLearning(
                estimator=self.modules_funcs.get(ModuleKind.BASEMODEL.value)(),
                hard_example_mining=self.modules_funcs.get(
                    ModuleKind.HARD_EXAMPLE_MINING.value)())

        return None
