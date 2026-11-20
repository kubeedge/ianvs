# Copyright 2026 The KubeEdge Authors.
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

"""Rank picture mode reads the matrix metric under the key it is stored as
(core/storymanager/rank/rank.py).

TestCase.compute_metrics keys each result by the metric name from testenv.yaml,
and only the SystemMetricType name "MATRIX" resolves to matrix_func, so the
heatmaps have to be looked up under "MATRIX".
"""

import types
import uuid

from core.common.constant import SystemMetricType
from core.storymanager.rank import rank as rank_module
from core.storymanager.rank.rank import Rank


def test_draw_pictures_uses_matrix_metric_key(tmp_path, monkeypatch):
    """Each matrix in the MATRIX result is drawn into the test case output dir."""
    drawn = []
    monkeypatch.setattr(rank_module, "draw_heatmap_picture",
                        lambda output, title, matrix: drawn.append((output, title, matrix)))

    test_case = types.SimpleNamespace(id=uuid.uuid1(), output_dir=str(tmp_path))
    matrices = {"all": [[0.5, 0.6], [0.7, 0.8]], "task_avg": [[0.4, 0.5], [0.6, 0.7]]}
    test_results = {
        test_case.id: ({"accuracy": 0.9, SystemMetricType.MATRIX.value: matrices}, "time")
    }

    Rank({"sort_by": [{"accuracy": "descend"}]})._draw_pictures([test_case], test_results)

    assert drawn == [
        (str(tmp_path), "all", matrices["all"]),
        (str(tmp_path), "task_avg", matrices["task_avg"]),
    ]
