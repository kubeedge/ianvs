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

from collections import OrderedDict
from pathlib import Path

import motmetrics as mm
from loguru import logger
from sedna.common.class_factory import ClassType, ClassFactory

__all__ = ["f1_score"]


def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:
            logger.info('Comparing {}...'.format(k))
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
            names.append(k)
        else:
            logger.warning('No ground truth for {}, skipping.'.format(k))

    return accs, names

@ClassFactory.register(ClassType.GENERAL, alias="f1_score")
def metric(y_true, y_pred):
    gt = OrderedDict([(Path(f).parts[-3], mm.io.loadtxt(f, fmt='mot15-2D', min_confidence=1)) for f in y_true])
    mh = mm.metrics.create()
    accs, names = compare_dataframes(gt, y_pred)

    logger.info('Running metrics')
    metrics_name = ['precision', 'recall']
    summary = mh.compute_many(accs, names=names, metrics=metrics_name, generate_overall=True)
    logger.info('Completed')

    precision = float(summary.iloc[-1][['precision']])
    recall = float(summary.iloc[-1][['recall']])
    return round(2*((precision*recall)/(precision+recall)), 4)
=======
version https://git-lfs.github.com/spec/v1
oid sha256:fe84c202e412373657e855856b14e644f7b0a2c2ac3f0890076268b14bbd8998
size 1835
>>>>>>> 9676c3e (ya toh aar ya toh par)
