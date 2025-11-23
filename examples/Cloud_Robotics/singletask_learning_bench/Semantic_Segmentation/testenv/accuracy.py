<<<<<<< HEAD
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
from tqdm import tqdm

from sedna.common.class_factory import ClassType, ClassFactory

from RFNet.dataloaders import make_data_loader
from RFNet.utils.metrics import Evaluator
from RFNet.utils.args import ValArgs

__all__ = ('accuracy', 'compute_cpa', 'compute_miou', 'compute_fwiou')

logger = logging.getLogger(__name__)

@ClassFactory.register(ClassType.GENERAL, alias="accuracy")
def accuracy(y_true, y_pred, **kwargs):
    args = ValArgs()
    _, _, test_loader, num_class = make_data_loader(args, test_data=y_true)
    evaluator = Evaluator(num_class)
    tbar = tqdm(test_loader, desc='\r')
    for i, (sample, img_path) in enumerate(tbar):
        if args.depth:
            image, depth, target = sample['image'], sample['depth'], sample['label']
        else:
            image, target = sample['image'], sample['label']
        if args.cuda:
            image, target = image.cuda(args.gpu_ids), target.cuda(args.gpu_ids)
            if args.depth:
                depth = depth.cuda(args.gpu_ids)

        target[target > evaluator.num_class-1] = 255
        target = target.cpu().numpy()
        evaluator.add_batch(target, y_pred[i])

    CPA = compute_cpa(evaluator)
    mIoU = compute_miou(evaluator)
    FWIoU = compute_fwiou(evaluator)

    logger.info("CPA:{}, mIoU:{}, fwIoU: {}".format(CPA, mIoU, FWIoU))
    return mIoU

def compute_cpa(evaluator):
    CPA = evaluator.Pixel_Accuracy_Class()
    logger.info(f"CPA: {CPA}")
    return CPA

def compute_miou(evaluator):
    mIoU = evaluator.Mean_Intersection_over_Union()
    logger.info(f"mIoU: {mIoU}")
    return mIoU

def compute_fwiou(evaluator):
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    logger.info(f"FWIoU: {FWIoU}")
    return FWIoU
=======
version https://git-lfs.github.com/spec/v1
oid sha256:a091c69d011ff86466d101675fd5295896fe170384c29e786030e5ec319032c7
size 2342
>>>>>>> 9676c3e (ya toh aar ya toh par)
