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

import numpy as np


def cmc(distmat, query_ids, gallery_ids, topk):
    m, _ = distmat.shape
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    ret = np.zeros(topk)
    for i in range(m):
        nonzero = np.nonzero(matches[i])[0]
        # Fix #447: skip queries with no matching identity in the gallery.
        # np.nonzero()[0][0] raised IndexError on empty arrays before this fix.
        if len(nonzero) == 0:
            continue
        k = nonzero[0]
        if k < topk:
            ret[k] += 1
    return round(float(ret.cumsum()[-1] / m), 4)


def rank_5(y_pred, y_true, **kwargs):
    return cmc(y_pred, y_true[:, 0], y_true[:, 1], topk=5)
