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

"""Unit tests for core.storymanager.rank.rank.Rank._get_all"""

import time
import unittest
from types import SimpleNamespace

from core.storymanager.rank.rank import Rank


def _build_rank():
    config = {
        "sort_by": [{"acc": "descend"}],
        "selected_dataitem": {
            "paradigms": ["all"],
            "modules": ["all"],
            "hyperparameters": ["all"],
            "metrics": ["all"],
        },
    }
    rank = Rank(config)
    rank.all_df_header = ["algorithm", "paradigm", "acc", "time", "url"]
    rank.all_rank_file = ""
    return rank


def _build_test_cases(n):
    test_cases = []
    test_results = {}
    for i in range(n):
        algorithm = SimpleNamespace(
            name=f"algorithm-{i}",
            paradigm_type="singletasklearning",
            modules={},
        )
        test_case = SimpleNamespace(
            id=i,
            algorithm=algorithm,
            output_dir=f"/tmp/out-{i}",
        )
        test_cases.append(test_case)
        test_results[i] = ({"acc": float(i)}, 0.1)
    return test_cases, test_results


# pylint: disable=protected-access,missing-function-docstring
class TestRankGetAll(unittest.TestCase):
    """Tests for Rank._get_all row construction."""

    def test_get_all_returns_expected_rows(self):
        rank = _build_rank()
        test_cases, test_results = _build_test_cases(5)

        all_df = rank._get_all(test_cases, test_results)

        self.assertEqual(len(all_df), 5)
        self.assertEqual(
            sorted(all_df["algorithm"].tolist()),
            sorted(f"algorithm-{i}" for i in range(5)),
        )

    def test_get_all_scales_near_linearly(self):
        rank = _build_rank()

        small_cases, small_results = _build_test_cases(200)
        start = time.time()
        rank._get_all(small_cases, small_results)
        small_duration = time.time() - start

        large_cases, large_results = _build_test_cases(2000)
        start = time.time()
        rank._get_all(large_cases, large_results)
        large_duration = time.time() - start

        # 10x the rows should cost far less than 10x^2 = 100x the time.
        # A generous ceiling of 30x guards against reintroducing the
        # quadratic `all_df.loc[i] = row_data` pattern while remaining
        # tolerant of CI machine variance.
        self.assertLess(large_duration, max(small_duration * 30, 1.0))


if __name__ == "__main__":
    unittest.main()
