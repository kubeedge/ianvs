import numpy as np

from sedna.common.class_factory import ClassFactory, ClassType

@ClassFactory.register(ClassType.GENERAL, alias="OracleRouterDatasetProcessor")
class OracleRouterDatasetProcessor:
    def __init__(self, **kwargs):
        pass

    def __call__(self, dataset):
        dataset.x = [{"query": x, "gold": y}  for x,y in zip(dataset.x, dataset.y)]
        return dataset