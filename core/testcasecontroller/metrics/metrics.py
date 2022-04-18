import sys
import numpy as np

from sedna.common.class_factory import ClassFactory, ClassType

from core.common.utils import load_module

def smape(y_true, y_pred):
    y_true = np.array(y_true).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)
    return np.mean(np.nan_to_num(np.abs(y_true - y_pred) / (np.abs(y_pred) + np.abs(y_true))))


def max_error_rate(y_true, y_pred):
    return max(np.nan_to_num(np.abs(y_true - y_pred) / (np.abs(y_pred) + np.abs(y_true))))


def get_metric_func(metric_name: str = None, metric_dict: dict = None):
    """ get metric func """
    if isinstance(metric_name, str):
        return getattr(sys.modules[__name__], metric_name)
    elif isinstance(metric_dict, dict):
        name = metric_dict.get("name")
        url = metric_dict.get("url")
        if url:
            load_module(url)
            try:
                metric_func = ClassFactory.get_cls(type_name=ClassType.GENERAL, t_cls_name=name)
            except Exception as err:
                raise Exception(f"get metric func(url={url}) failed, error: {err}.")
            return metric_func
        else:
            return getattr(sys.modules[__name__], name)
