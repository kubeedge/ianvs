import json

from sedna.common.class_factory import ClassType, ClassFactory
__all__ = ["map"]


@ClassFactory.register(ClassType.GENERAL, alias="map")
def map(y_true, y_pred, **kwargs):
    print("loading test")
