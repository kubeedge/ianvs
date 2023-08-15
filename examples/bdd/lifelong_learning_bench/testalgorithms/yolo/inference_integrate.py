# Copyright 2021 The KubeEdge Authors.
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

"""
Integrate the inference results of all related tasks
"""

from re import L
from typing import List

import numpy as np
import torch
from sedna.common.class_factory import ClassFactory, ClassType


class Task:
    def __init__(self, entry, samples, meta_attr=None):
        self.entry = entry
        self.samples = samples
        self.meta_attr = meta_attr
        self.test_samples = None  # assign on task definition and use in TRD
        self.model = None  # assign on running
        self.result = None  # assign on running


__all__ = ("InferenceIntegrate",)

yolo_hub_path = "/home/shifan/.cache/torch/hub/ultralytics_yolov5_master"


@ClassFactory.register(ClassType.STP)
class InferenceIntegrate:
    """
    Default calculation algorithm for inference integration

    Parameters
    ----------
    models: All models used for sample inference
    """

    def __init__(self, **kwargs):
        self.model = None

    def load(self, model_url, **kwargs):
        if model_url:
            self.model = torch.hub.load(
                yolo_hub_path, "custom", path=model_url, source="local"
            )
        else:
            raise Exception("model url does not exist.")

    def predict(self, data, **kwargs):
        if type(data) is np.ndarray:
            data = data.tolist()
        with_nms, model_forward_result = kwargs.get("with_nms"), kwargs.get(
            "model_forward_result"
        )
        only_nms, conf = kwargs.get("only_nms"), kwargs.get("conf")
        self.model.eval()
        predictions = []
        if not with_nms:
            result = self.model(data, with_nms=with_nms, size=640)
            return result
        else:
            result = self.model(
                data,
                model_forward_result=model_forward_result,
                only_nms=only_nms,
                conf=conf,
            )
            predictions.append(np.array(result.pandas().xywhn[0]))
            return predictions

    def __call__(self, tasks: List[Task]):
        """
        Parameters
        ----------
        tasks: All tasks with sample result

        Returns
        -------
        result: minimum result
        """
        pred_l = []
        for task in tasks:
            pred_l.append(task.result[0])
        pred_l = torch.stack(pred_l)
        pred_l = pred_l.unsqueeze(0)
        kwargs = {
            "with_nms": True,
            "model_forward_result": pred_l,
            "only_nms": True,
            "conf": 0.6,
        }
        self.load(task.model.model)
        res = self.predict(task.samples, **kwargs)
        return res[0]
