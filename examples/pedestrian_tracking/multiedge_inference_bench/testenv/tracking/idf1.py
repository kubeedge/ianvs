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

__all__ = ["idf1"]


def compare_dataframes(gts, ts):
    accs = []
    names = []
    for k, tsacc in ts.items():
        if k in gts:
            logger.info(f"Comparing {k}...")
            accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, "iou", distth=0.5))
            names.append(k)
        else:
            logger.warning(f"No ground truth for {k}, skipping.")

    return accs, names


@ClassFactory.register(ClassType.GENERAL, alias="idf1")
def metric(y_true, y_pred):
    gt = OrderedDict(
        [(Path(f).parts[-3], mm.io.loadtxt(f, fmt="mot15-2D", min_confidence=1)) for f in y_true]
    )
    mh = mm.metrics.create()
    accs, names = compare_dataframes(gt, y_pred)

    logger.info("Running metrics")
    metrics_name = ["idf1"]
    summary = mh.compute_many(accs, names=names, metrics=metrics_name, generate_overall=True)
    logger.info("Completed")

    return round(float(summary.iloc[-1][metrics_name]), 4)
