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

import sys
import unittest
from unittest.mock import MagicMock

# Mock third-party imports before loading rank module
sys.modules['numpy'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['core.storymanager.visualization'] = MagicMock()

from core.storymanager.rank.rank import Rank


class TestRankSortAllDf(unittest.TestCase):
    """Unit tests for Rank._sort_all_df method."""

    def test_sort_with_dict_elements(self):
        rank = Rank({"sort_by": [{"accuracy": "descend"}]})
        mock_df = MagicMock()
        mock_df.columns = ["accuracy", "algorithm"]
        
        res = rank._sort_all_df(mock_df, ["accuracy"])
        mock_df.sort_values.assert_called_once_with(by=["accuracy"], ascending=[False])

    def test_sort_with_string_elements(self):
        rank = Rank({"sort_by": ["accuracy"]})
        mock_df = MagicMock()
        mock_df.columns = ["accuracy", "algorithm"]
        
        res = rank._sort_all_df(mock_df, ["accuracy"])
        mock_df.sort_values.assert_called_once_with(by=["accuracy"], ascending=[True])

    def test_sort_with_empty_sort_metric_list(self):
        rank = Rank({"sort_by": [{"non_existent": "ascend"}]})
        mock_df = MagicMock()
        mock_df.columns = ["accuracy", "algorithm"]
        
        res = rank._sort_all_df(mock_df, ["accuracy"])
        self.assertEqual(res, mock_df)
        mock_df.sort_values.assert_not_called()

    def test_sort_by_dataframe_column(self):
        rank = Rank({"sort_by": [{"algorithm": "ascend"}]})
        mock_df = MagicMock()
        mock_df.columns = ["accuracy", "algorithm"]
        
        res = rank._sort_all_df(mock_df, ["accuracy"])
        mock_df.sort_values.assert_called_once_with(by=["algorithm"], ascending=[True])


if __name__ == "__main__":
    unittest.main()
