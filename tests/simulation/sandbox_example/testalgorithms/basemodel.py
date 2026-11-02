"""
A threshold classifier for testing the simulation sandbox. The `mode` hyperparameter
makes a test case succeed (ok), exceed its memory limit (oom) or raise an error (fail).
"""

import os

import numpy as np
from sedna.common.class_factory import ClassType, ClassFactory

__all__ = ["BaseModel"]


@ClassFactory.register(ClassType.GENERAL, alias="ThresholdClassifier")
class BaseModel:
    """predicts 1 when x is above the midpoint of the two class means."""

    def __init__(self, **kwargs):
        self.mode = kwargs.get("mode", "ok")
        self.threshold = 0.0

    def train(self, train_data, valid_data=None, **kwargs):
        """learn the threshold from the samples x and their labels y."""
        x, y = np.asarray(train_data.x, dtype=float), np.asarray(train_data.y, dtype=int)
        if self.mode == "fail":
            raise ValueError("training failed on purpose (mode=fail)")
        if self.mode == "oom":
            # touch 4 GiB, more than the paradigm's default memory limit
            np.ones((4 * 1024, 1024 * 1024 // 8))
        self.threshold = float((x[y == 0].mean() + x[y == 1].mean()) / 2)

    def save(self, model_path):
        """write the threshold; returns the model file."""
        os.makedirs(model_path, exist_ok=True)
        model_file = os.path.join(model_path, "threshold.txt")
        with open(model_file, "w", encoding="utf-8") as file:
            file.write(str(self.threshold))
        return model_file

    def load(self, model_url=None):
        """read the threshold."""
        with open(model_url, encoding="utf-8") as file:
            self.threshold = float(file.read())

    def predict(self, data, **kwargs):
        """the predicted labels."""
        return (np.asarray(data, dtype=float) > self.threshold).astype(int)
