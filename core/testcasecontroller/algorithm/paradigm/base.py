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

from core.common.constant import ModuleType, ParadigmType


class ParadigmBase:
    """
    Paradigm Base
    Notes:
          1. Ianvs serves as testing tools for test objects, e.g., algorithms.
          2. Ianvs does NOT include code directly on test object.
          3. Algorithms serve as typical test objects in Ianvs
          and detailed algorithms are thus NOT included in this Ianvs python file.
          4. As for the details of example test objects, e.g., algorithms,
          please refer to third party packages in Ianvs example.
          For example, AI workflow and interface pls refer to sedna
          (sedna docs: https://sedna.readthedocs.io/en/latest/api/lib/index.html),
          and module implementation pls refer to `examples' test algorithms`,
          e.g., basemodel.py, hard_example_mining.py.

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
        for module_type, module in self.modules.items():
            func = module.get_module_func(module_type)
            if callable(func):
                module_funcs.update({module_type: func})
        return module_funcs

    def build_paradigm_job(self, paradigm_type):
        """
        build paradigm job instance according to paradigm type.
        this job instance provides the test flow of some algorithm modules.

        Parameters
        ---------
        paradigm_type: str

        Returns
        -------
        instance

        """
        if paradigm_type == ParadigmType.SINGLE_TASK_LEARNING.value:
            return self.modules_funcs.get(ModuleType.BASEMODEL.value)()

        if paradigm_type == ParadigmType.INCREMENTAL_LEARNING.value:
            return IncrementalLearning(
                estimator=self.modules_funcs.get(ModuleType.BASEMODEL.value)(),
                hard_example_mining=self.modules_funcs.get(
                    ModuleType.HARD_EXAMPLE_MINING.value)())

        return None
