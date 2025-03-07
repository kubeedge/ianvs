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

""" Algorithm Module"""

import copy

from sedna.common.class_factory import ClassFactory, ClassType

from core.common import utils
from core.common.constant import ModuleType
from core.testcasecontroller.generation_assistant import get_full_combinations


# pylint: disable=too-few-public-methods
class Module:
    """
    Algorithm Module:
    provide the configuration and the calling functions of the algorithm module.
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
    ----------
    config : dict
         config of the algorithm module, includes: type, name,
         url of the python file that defines algorithm module,
         hyperparameters of the calling functions of algorithm module, etc.

    """

    def __init__(self, config):
        self.type: str = ""
        self.name: str = ""
        self.url: str = ""
        self.hyperparameters = {}
        self.hyperparameters_list = []
        self._parse_config(config)

    def _check_fields(self):
        if not self.type and not isinstance(self.type, str):
            raise ValueError(f"module type({self.type}) must be provided and be string type.")

        types = [e.value for e in ModuleType.__members__.values()]
        if self.type not in types:
            raise ValueError(f"not support module type({self.type}."
                             f"the following paradigms can be selected: {types}")

        if not self.name and not isinstance(self.name, str):
            raise ValueError(f"module name({self.name}) must be provided and be string type.")

        if not isinstance(self.url, str):
            raise ValueError(f"module url({self.url}) must be string type.")

    #pylint: disable=too-many-branches
    def get_module_instance(self, module_type):
        """
        get function of algorithm module by using module type

        Parameters
        ---------
        module_type: string
            module type, e.g.: basemodel, hard_example_mining, etc.

        Returns
        ------
        function

        """
        class_factory_type = ClassType.GENERAL
        if module_type in [ModuleType.HARD_EXAMPLE_MINING.value]:
            class_factory_type = ClassType.HEM

        elif module_type in [ModuleType.TASK_DEFINITION.value,
                             ModuleType.TASK_RELATIONSHIP_DISCOVERY.value,
                             ModuleType.TASK_REMODELING.value,
                             ModuleType.TASK_ALLOCATION.value,
                             ModuleType.INFERENCE_INTEGRATE.value]:
            class_factory_type = ClassType.STP

        elif module_type in [ModuleType.TASK_UPDATE_DECISION.value]:
            class_factory_type = ClassType.KM

        elif module_type in [ModuleType.UNSEEN_TASK_ALLOCATION.value]:
            class_factory_type = ClassType.UTP

        elif module_type in [ModuleType.UNSEEN_SAMPLE_RECOGNITION.value,
                             ModuleType.UNSEEN_SAMPLE_RE_RECOGNITION.value]:
            class_factory_type = ClassType.UTD
        elif module_type in [ModuleType.AGGREGATION.value]:
            class_factory_type = ClassType.FL_AGG
            agg = None
            if self.url :
                try:
                    utils.load_module(self.url)
                    agg = ClassFactory.get_cls(
                        type_name=class_factory_type, t_cls_name=self.name)(**self.hyperparameters)
                except Exception as err:
                    raise RuntimeError(f"module(type={module_type} loads class(name={self.name}) "
                                    f"failed, error: {err}.") from err
            return self.name, agg

        if self.url:
            try:
                utils.load_module(self.url)

                if class_factory_type == ClassType.HEM:
                    func = {"method": self.name, "param":self.hyperparameters}
                else:
                    func = ClassFactory.get_cls(
                        type_name=class_factory_type,
                        t_cls_name=self.name
                    )(**self.hyperparameters)

                return func

            except Exception as err:
                raise RuntimeError(f"module(type={module_type} loads class(name={self.name}) "
                                f"failed, error: {err}.") from err

        # call lib built-in module function
        module_func = {"method": self.name}
        if self.hyperparameters:
            module_func["param"] = self.hyperparameters

        return module_func

    def _parse_config(self, config):
        # pylint: disable=C0103
        for k, v in config.items():
            if k == "hyperparameters":
                self.hyperparameters_list = self._parse_hyperparameters(v)
            if k in self.__dict__:
                self.__dict__[k] = v

        self._check_fields()

    def _parse_hyperparameters(self, config):
        # hp is short for hyperparameters
        base_hps = {}
        hp_name_values_list = []
        for ele in config:
            hp_config = ele.popitem()
            hp_name = hp_config[0]
            hp_values = hp_config[1].get("values")
            if hp_name == "other_hyperparameters":
                base_hps = self._parse_other_hyperparameters(hp_values)
            else:
                hp_name_values_list.append((hp_name, hp_values))

        hp_combinations_list = get_full_combinations(hp_name_values_list)

        hps_list = []
        for hp_combinations in hp_combinations_list:
            base_hps_copy = copy.deepcopy(base_hps)
            base_hps_copy.update(**hp_combinations)
            hps_list.append(base_hps_copy)

        return hps_list

    @classmethod
    def _parse_other_hyperparameters(cls, config_files):
        base_hps = {}
        for hp_config_file in config_files:
            if not utils.is_local_file(hp_config_file):
                raise RuntimeError(f"not found other hyperparameters config file"
                                f"({hp_config_file}) in local")

            try:
                other_hps = utils.yaml2dict(hp_config_file)
                base_hps.update(**other_hps)
            except Exception as err:
                raise RuntimeError(
                    f"other hyperparameters config file({hp_config_file}) is unvild, "
                    f"error: {err}") from err
        return base_hps
