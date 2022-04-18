from sedna.common.class_factory import ClassFactory, ClassType

from core.common import utils
from core.common.constant import ParadigmKind


class Module:
    def __init__(self):
        self.kind: str = ""
        self.name: str = ""
        self.url: str = ""
        self.hyperparameters: dict = {}

    def check_fields(self):
        if not self.kind and not isinstance(self.kind, str):
            raise ValueError(f"the field of module kind({self.kind}) is unvaild.")
        if not self.name and not isinstance(self.name, str):
            raise ValueError(f"the field of module name({self.name}) is unvaild.")
        if not isinstance(self.url, str):
            raise ValueError(f"the field of module url({self.url}) is unvaild.")
        if not isinstance(self.hyperparameters, dict):
            raise ValueError(f"the field of module hyperparameters({self.hyperparameters}) is unvaild.")


class Algorithm:
    def __init__(self):
        self.name: str = ""
        self.paradigm: str = ""
        self.dataset_train_ratio: float = 0.8
        self.initial_model_url: str = ""
        self.modules: dict = {}

    def check_fields(self):
        if not self.name and not isinstance(self.name, str):
            raise ValueError(f"the field of algorithm name({self.name}) is unvaild.")
        if not self.paradigm and not isinstance(self.paradigm, str):
            raise ValueError(f"the field of algorithm paradigm({self.paradigm}) is unvaild.")
        if not isinstance(self.dataset_train_ratio, float):
            raise ValueError(f"the field of algorithm dataset_train_ratio({self.dataset_train_ratio}) is unvaild.")
        if not isinstance(self.initial_model_url, str):
            raise ValueError(f"the field of algorithm initial_model_url({self.initial_model_url}) is unvaild.")
        for m in self.modules:
            m.check_fields()

    def build(self):
        feature_process = None
        basemodel_object = None
        for module in self.modules:
            if module.kind == "basemodel":
                self.basemodel = module
                basemodel_object = self._get_basemodel(module)
            elif module.kind == "feature_process":
                feature_process = self._get_feature_process_func(module)
        job = basemodel_object

        if self.paradigm == ParadigmKind.IncrementalLearning.value:
            from sedna.core.incremental_learning import IncrementalLearning
            job = IncrementalLearning(estimator=basemodel_object)

        return job, feature_process

    def _get_basemodel(self, module: Module):
        if not module.url and isinstance(module.url, str):
            raise ValueError(f"the field of module({module.kind}) url({module.url}) is unvaild.")

        utils.load_module(module.url)
        try:
            basemodel = ClassFactory.get_cls(type_name=ClassType.GENERAL, t_cls_name=module.name)(
                **module.hyperparameters)
        except Exception as err:
            raise Exception(f"basemodel module loads class(name={module.name}) failed, error: {err}.")

        return basemodel

    def _get_feature_process_func(self, module: Module):
        if not module.url and isinstance(module.url, str):
            raise ValueError(f"the field of module({module.kind}) url({module.url}) is unvaild.")

        utils.load_module(module.url)
        try:
            features_process_func = ClassFactory.get_cls(type_name=ClassType.GENERAL,
                                                         t_cls_name=module.name)
        except Exception as err:
            raise Exception(f"feature_process module loads function(name={module.name}) failed, error: {err}.")

        return features_process_func
