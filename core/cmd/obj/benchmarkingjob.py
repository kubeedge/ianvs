import copy
import os
from itertools import product

from core.testcasecontroller.algorithm import Algorithm, Module
from core.testcasecontroller.testcase import TestCase
from core.storymanager.rank import Rank
from core.common import utils
from core.testenvmanager.testenv import TestEnv


class TestJob:
    def __init__(self, config):
        self.name: str = ""
        self.workspace: str = "./workspace"
        self.testenv: TestEnv = TestEnv()
        self.metrics: list = []
        self.algorithms: list = []
        self.rank: Rank = Rank()
        self._parse_config(config)

    def _check_fields(self):
        if not self.name and not isinstance(self.name, str):
            raise ValueError(f"the field of testjob name({self.name}) is unvaild.")
        if not isinstance(self.workspace, str):
            raise ValueError(f"the field of testjob workspace({self.workspace}) is unvaild.")
        if not isinstance(self.metrics, list):
            raise ValueError(f"the field of testjob metrics({self.metrics}) is unvaild.")

        self.testenv.check_fields()
        for algorithm in self.algorithms:
            algorithm.check_fields()
        self.rank.check_fields()

    def run(self):
        try:
            self.workspace = os.path.join(self.workspace, self.name)
            if not self.metrics:
                self.metrics = self.testenv.metrics
            self.testenv.prepare(self.workspace)
        except Exception as err:
            raise Exception(f"prepare test env failed, error: {err}.")

        test_cases = []
        test_results = {}
        try:
            for algorithm in self.algorithms:
                test_cases.append(TestCase(self.testenv, algorithm))
            for test_case in test_cases:
                test_case.prepare(self.metrics, self.workspace)
                test_results[test_case.id] = (test_case.run(), utils.get_local_time())
        except Exception as err:
            raise Exception(f"test cases run failed, error:{err}.")
        if test_results:
            try:
                self.rank.save(test_cases, test_results, output_dir=self.workspace)
            except Exception as err:
                raise Exception(f"test job(name={self.name}) saves test results failed, error: {err}.")

            try:
                self.rank.plot()
            except Exception as err:
                raise Exception(f"test job(name={self.name}) plots test results failed, error: {err}")

    def _parse_config(self, config: dict):
        try:
            for k, v in config.items():
                if k == "testenv":
                    self._parse_testenv_config(v, self.testenv)
                elif k == str.lower(Algorithm.__name__ + "s"):
                    self._parse_algorithms_config(v, self.algorithms)
                elif k == str.lower(Rank.__name__):
                    self._parse_rank_config(v, self.rank)
                else:
                    if k in self.__dict__.keys():
                        self.__dict__[k] = v
            self._check_fields()
        except Exception as err:
            raise ValueError(f"parse testjob config file failed, error: {err}.")

    def _parse_testenv_config(self, file, testenv):
        if os.path.isfile(file) and os.path.exists(file):
            config = utils.yaml2dict(file)
            testenv_config = config[str.lower(TestEnv.__name__)]
        else:
            raise Exception(f"test env({file}) is unvaild")

        for k, v in testenv_config.items():
            if k == "dataset":
                self._dict2object(v, testenv.dataset)
            else:
                if k in testenv.__dict__.keys():
                    testenv.__dict__[k] = v

    def _parse_rank_config(self, config: dict, rank):
        for k, v in config.items():
            if k in rank.__dict__.keys():
                rank.__dict__[k] = v

    def _dict2object(self, dict, obj):
        for k, v in dict.items():
            if k in obj.__dict__.keys():
                obj.__dict__[k] = v
        return obj

    def _parse_algorithms_config(self, config: list, algorithms):
        names = []
        for algorithm_config in config:
            algorithm = Algorithm()
            name = algorithm_config.get("name")
            if name not in names:
                algorithm.name = name
                url = algorithm_config.get("url")
                algorithm_list = self._parse_algorithm_file(url, algorithm)
                algorithms.extend(algorithm_list)
            else:
                raise Exception(f"algorithm name({name}) is not unique.")

    def _parse_algorithm_file(self, file, algorithm: Algorithm):
        if os.path.isfile(file) and os.path.exists(file):
            config = utils.yaml2dict(file)
            algorithm_config = config[str.lower(Algorithm.__name__)]
        else:
            raise Exception(f"algorithm url({file}) is unvaild")

        for k, v in algorithm_config.items():
            if k in algorithm.__dict__.keys() and k != str.lower(Module.__name__ + "s"):
                algorithm.__dict__[k] = v

        modules_config = algorithm_config.get(str.lower(Module.__name__ + "s"))
        modules_list = None
        if modules_config:
            modules_list = self._parse_modules_config(modules_config)

        algorithm_list = []
        for module_list in modules_list:
            a = copy.deepcopy(algorithm)
            a.modules = module_list
            algorithm_list.append(a)
        del algorithm

        return algorithm_list

    def _parse_modules_config(self, config: list):
        modules_list = []
        for module_config in config:
            module_list = []
            module = Module()
            for k, v in module_config.items():
                if k in module.__dict__.keys() and k != "hyperparameters":
                    module.__dict__[k] = v

            hps = module_config.get("hyperparameters")
            if hps:
                new_hps_list = self._parse_hyperparameters_config(hps)
                for new_hps in new_hps_list:
                    m = copy.deepcopy(module)
                    m.hyperparameters = new_hps
                    module_list.append(m)
                del module
            else:
                module_list.append(module)
            modules_list.append(module_list)
        return product(*modules_list)

    def _parse_hyperparameters_config(self, config: list):
        base_hps = {}
        name_list = []
        values_list = []
        for ele in config:
            hp = ele.popitem()
            name = hp[0]
            values = hp[1].get("values")
            if name == "other_hyperparameters":
                for value in values:
                    other_hps = utils.yaml2dict(value)
                    base_hps = {**base_hps, **other_hps}
            else:
                name_list.append(name)
                values_list.append(values)

        new_hps_list = []
        for new_hps_combinations in product(*values_list):
            hps = copy.deepcopy(base_hps)
            for i in range(len(name_list)):
                hps[name_list[i]] = new_hps_combinations[i]
            new_hps_list.append(copy.deepcopy(hps))
        return new_hps_list
