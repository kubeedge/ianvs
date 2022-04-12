import copy
from itertools import product

from .testenv import TestEnv
from .algorithm import Algorithm, Paradigm, BaseModel


class TestJob:
    def __init__(self, config):
        self.name = ""
        self.test_env = TestEnv()
        self.algorithm_list = []
        self.parse(config)

    def parse(self, test_job_config):
        for key in test_job_config:
            v = test_job_config[key]
            if key == str.lower(TestEnv.__name__):
                self.dict_to_object(self.test_env, v)
            elif key == str.lower(Algorithm.__name__):
                self.parse_algorithms_config(v)
            else:
                self.__dict__[key] = test_job_config[key]

    def parse_algorithms_config(self, algorithms_config):
        def parse_basemodels_config(basemodels_config):
            basemodel_list = []
            for b in basemodels_config:
                parameters_list = []
                for parameter in b.multi_parameters:
                    parameters_list.append(b.hyperparameters[parameter])
                for parameter_list in product(*parameters_list):
                    basemodel = copy.deepcopy(b)
                    for i in range(len(b.multi_parameters)):
                        basemodel.hyperparameters[b.multi_parameters[i]] = parameter_list[i]
                    basemodel_list.append(basemodel)
            return basemodel_list

        def parse_algorithms_config(key, cls):
            objects = []
            configs = algorithms_config[key]
            if len(configs) > 0:
                for config in configs:
                    obj = cls()
                    self.dict_to_object(obj, config)
                    objects.append(obj)
            if cls == BaseModel:
                return parse_basemodels_config(objects)
            return objects

        paradigms = parse_algorithms_config(str.lower(Paradigm.__name__) + "s", Paradigm)
        basemodels = parse_algorithms_config(str.lower(BaseModel.__name__) + "s", BaseModel)

        for p in paradigms:
            for b in basemodels:
                algorithm = Algorithm()
                algorithm.paradigm = p
                algorithm.basemodel = b
                self.algorithm_list.append(algorithm)

    def dict_to_object(self, obj, object_config):
        for k in object_config:
            obj.__dict__[k] = object_config[k]
