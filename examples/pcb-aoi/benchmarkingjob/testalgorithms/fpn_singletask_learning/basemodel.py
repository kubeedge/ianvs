import os
from inspect import getfullargspec

from FPN_TensorFlow.interface import Estimator as Model
from sedna.common.class_factory import ClassType, ClassFactory

os.environ['BACKEND_TYPE'] = 'TENSORFLOW'

__all__ = ["BaseModel"]


def parse_kwargs(func, **kwargs):
    """ get valid parameters in kwargs """
    if not callable(func):
        return kwargs
    need_kw = getfullargspec(func)
    if need_kw.varkw == 'kwargs':
        return kwargs
    return {k: v for k, v in kwargs.items() if k in need_kw.args}

@ClassFactory.register(ClassType.GENERAL, "estimator")
class BaseModel:
    def __init__(self, **kwargs):
        varkw = parse_kwargs(Model, **kwargs)
        self.model = Model(**varkw)

    def train(self, train_data, valid_data=None, **kwargs):
        return self.model.train(train_data, **kwargs)

    def predict(self, data, **kwargs):
        # data -> image urls
        return self.model.predict(data, **kwargs)

    def load(self, model_url):
        self.model.load(model_url)

    def save(self, model_path):
        return self.model.save(model_path)

    def evaluate(self, data, **kwargs):
        return self.model.evaluate(data, **kwargs)