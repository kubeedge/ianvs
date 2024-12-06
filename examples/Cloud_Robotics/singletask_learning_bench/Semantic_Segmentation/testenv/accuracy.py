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

from tqdm import tqdm

from sedna.common.class_factory import ClassType, ClassFactory

from RFNet.dataloaders import make_data_loader
from RFNet.utils.metrics import Evaluator
from RFNet.utils.args import ValArgs

__all__ = ('accuracy')


@ClassFactory.register(ClassType.GENERAL, alias="accuracy")
def accuracy(y_true, y_pred, **kwargs):
    args = ValArgs()
    _, _, test_loader, num_class = make_data_loader(args, test_data=y_true)
    evaluator = Evaluator(num_class)
    #print(y_true)
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
        #if i% 20 == 0:
        #    print(img_path)
        #    print(image, target, y_pred[i])
        # Add batch sample into evaluator
        evaluator.add_batch(target, y_pred[i])

    # Test during the training
    # Acc = evaluator.Pixel_Accuracy()
    CPA = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

    print("CPA:{}, mIoU:{}, fwIoU: {}".format(CPA, mIoU, FWIoU))
    return mIoU