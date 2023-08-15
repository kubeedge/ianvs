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

import numpy as np
from sedna.common.class_factory import ClassFactory, ClassType

__all__ = ["rank_5"]


def cmc(distmat, query_ids, gallery_ids, topk):
    m, _ = distmat.shape
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = gallery_ids[indices] == query_ids[:, np.newaxis]
    # Compute CMC for each query
    ret = np.zeros(topk)
    for i in range(m):
        k = np.nonzero(matches[i])[0][0]
        if k < topk:
            ret[k] += 1
    return round(float(ret.cumsum()[-1] / m), 4)


@ClassFactory.register(ClassType.GENERAL, alias="rank_5")
def rank_5(query_ids, pred):
    query_ids = np.asarray([int(y.split("/")[-1]) for y in query_ids])
    distmat, gallery_ids = pred
    return cmc(distmat, query_ids, gallery_ids, 5)
