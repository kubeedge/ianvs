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
from core.common.log import LOGGER


def mean_ap(distmat, query_ids, gallery_ids):
    m, _ = distmat.shape
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    aps = []
    for i in range(m):
        valid = matches[i]
        if not valid.any():
            # No gallery match for this query — skip gracefully.
            continue
        tps = np.cumsum(valid)
        precision_at_k = tps / (np.arange(len(valid)) + 1)
        ap = (precision_at_k * valid).sum() / valid.sum()
        aps.append(ap)

    if len(aps) == 0:
        # Fix #447: replaced RuntimeError("No valid query") with a warning
        # and graceful 0.0 return so the pipeline does not crash.
        LOGGER.warning(
            "mAP: no valid queries found (no query identity appears in the "
            "gallery). Returning mAP=0.0. Check that query/gallery splits "
            "share at least one common identity."
        )
        return 0.0

    return round(float(np.mean(aps)), 4)


def mAP(y_pred, y_true, **kwargs):
    return mean_ap(y_pred, y_true[:, 0], y_true[:, 1])
