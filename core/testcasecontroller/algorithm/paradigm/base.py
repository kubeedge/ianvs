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
from sedna.core.lifelong_learning import LifelongLearning

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
        self.module_instances = self._get_module_instances()
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

    def _get_module_instances(self):
        module_instances = {}
        for module_type, module in self.modules.items():
            func = module.get_module_instance(module_type)
            module_instances.update({module_type: func})
        return module_instances

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
            return self.module_instances.get(ModuleType.BASEMODEL.value)

        if paradigm_type == ParadigmType.INCREMENTAL_LEARNING.value:
            return IncrementalLearning(
                estimator=self.module_instances.get(ModuleType.BASEMODEL.value),
                hard_example_mining=self.module_instances.get(
                    ModuleType.HARD_EXAMPLE_MINING.value))

        if paradigm_type == ParadigmType.LIFELONG_LEARNING.value:
            return LifelongLearning(
                estimator=self.module_instances.get(
                    ModuleType.BASEMODEL.value),
                task_definition=self.module_instances.get(
                    ModuleType.TASK_DEFINITION.value),
                task_relationship_discovery=self.module_instances.get(
                    ModuleType.TASK_RELATIONSHIP_DISCOVERY.value),
                task_allocation=self.module_instances.get(
                    ModuleType.TASK_ALLOCATION.value),
                task_remodeling=self.module_instances.get(
                    ModuleType.TASK_REMODELING.value),
                inference_integrate=self.module_instances.get(
                    ModuleType.INFERENCE_INTEGRATE.value),
                task_update_decision=self.module_instances.get(
                    ModuleType.TASK_UPDATE_DECISION.value),
                unseen_task_allocation=self.module_instances.get(
                    ModuleType.UNSEEN_TASK_ALLOCATION.value),
                unseen_sample_recognition=self.module_instances.get(
                    ModuleType.UNSEEN_SAMPLE_RECOGNITION.value),
                unseen_sample_re_recognition=self.module_instances.get(
                    ModuleType.UNSEEN_SAMPLE_RE_RECOGNITION.value)
            )
        # pylint: disable=E1101
        if paradigm_type == ParadigmType.MULTIEDGE_INFERENCE.value:
            return self.module_instances.get(ModuleType.BASEMODEL.value)

        return None
