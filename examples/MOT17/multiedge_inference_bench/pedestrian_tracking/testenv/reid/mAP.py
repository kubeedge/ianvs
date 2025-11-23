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

from __future__ import absolute_import

import numpy as np
from sklearn.metrics import average_precision_score
from sedna.common.class_factory import ClassType, ClassFactory

__all__ = ["mAP"]


def mean_ap(distmat, query_ids, gallery_ids):
    m, _ = distmat.shape
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute AP for each query
    aps = []
    for i in range(m):
        y_true = matches[i]
        y_score = -distmat[i][indices[i]]
        if not np.any(y_true): 
            continue
        aps.append(average_precision_score(y_true, y_score))
    if len(aps) == 0:
        raise RuntimeError("No valid query")
    return round(float(np.mean(aps)), 4)


@ClassFactory.register(ClassType.GENERAL, alias="mAP")
def mAP(query_ids, pred):
    query_ids = np.asarray([int(y.split('/')[-1]) for y in query_ids])
    distmat, gallery_ids = pred
    return mean_ap(distmat, query_ids, gallery_ids)
=======
version https://git-lfs.github.com/spec/v1
oid sha256:9f94556779b7b4b6b2c802c36f7d3b45a1390781a38e05510c50cc694b4794d5
size 1593
>>>>>>> 9676c3e (ya toh aar ya toh par)
