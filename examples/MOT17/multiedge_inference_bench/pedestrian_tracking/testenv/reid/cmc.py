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

from __future__ import absolute_import

import datetime
from pathlib import Path

import numpy as np
import matplotlib.ticker as mtick
import seaborn as sns
from sklearn.metrics import average_precision_score
from sedna.common.class_factory import ClassType, ClassFactory

__all__ = ["cmc"]


def plot_cmc(distmat, query_ids, gallery_ids, topk):
    m, n = distmat.shape
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute CMC for each query
    ret = np.zeros(topk)
    for i in range(m):
        k = np.nonzero(matches[i])[0][0]
        if k < topk:
            ret[k] += 1
    val = [0] + [rate for rate in ret.cumsum() / m]
    cmc = sns.lineplot(data=val, legend=False)
    cmc.set_title("CMC")
    cmc.set_xlabel("Rank")
    cmc.set_ylabel("Matching Rates (%)")
    cmc.yaxis.set_major_formatter(mtick.PercentFormatter(1.00))
    cmc.set(xlim=(0, topk), ylim=(0, 1))
    fig = cmc.get_figure()
    output_dir = Path("./examples/pedestrian_tracking/multiedge_inference_bench/cmc")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = Path(output_dir, datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".png")
    fig.savefig(output_file)
    return output_file


@ClassFactory.register(ClassType.GENERAL, alias="cmc")
def cmc(query_ids, pred):
    query_ids = np.asarray([int(y.split('/')[-1]) for y in query_ids])
    distmat, gallery_ids = pred
    return plot_cmc(distmat, query_ids, gallery_ids, 5)
