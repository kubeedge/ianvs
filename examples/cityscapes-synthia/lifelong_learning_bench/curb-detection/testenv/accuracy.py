# Copyright 2022 The KubeEdge Authors.
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

import logging
import numpy as np
from PIL import Image

from sedna.common.class_factory import ClassType, ClassFactory

logger = logging.getLogger(__name__)

from RFNet.utils.metrics import Evaluator
from RFNet.utils.args import ValArgs

__all__ = ('accuracy')


@ClassFactory.register(ClassType.GENERAL, alias="accuracy")
def accuracy(y_true, y_pred, **kwargs):
    args = ValArgs()
    num_class = args.num_class
    evaluator = Evaluator(num_class)

    for i, label_path in enumerate(y_true):
        if i >= len(y_pred):
            break
        target = np.array(Image.open(label_path.rstrip()))
        target[target > evaluator.num_class - 1] = 255
        pred = np.array(y_pred[i])
        while pred.ndim > 2:
            pred = pred[0]
        evaluator.add_batch(target, pred)

    if evaluator.confusion_matrix.sum() == 0:
        logger.warning("Empty confusion matrix, returning 0.0")
        return 0.0

    CPA = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

    logger.info("CPA: %s, mIoU: %s, fwIoU: %s", CPA, mIoU, FWIoU)
    return CPA
