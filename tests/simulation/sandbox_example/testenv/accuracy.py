"""accuracy of the sandbox example"""

import numpy as np
from sedna.common.class_factory import ClassType, ClassFactory

__all__ = ["accuracy"]


@ClassFactory.register(ClassType.GENERAL, alias="accuracy")
def accuracy(y_true, y_pred):
    """share of predictions equal to the labels."""
    return float(np.mean(np.asarray(y_true, dtype=int) == np.asarray(y_pred, dtype=int)))
