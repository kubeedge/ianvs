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

import unittest
from unittest.mock import MagicMock
from core.storymanager.rank.rank import Rank


class TestRank(unittest.TestCase):
    """Unit tests for Rank module error handling and edge cases."""

    def test_draw_pictures_with_none_matrix(self):
        """Test _draw_pictures when test_result has None or missing Matrix."""
        config = {
            "sort_by": [{"accuracy": "descend"}],
            "visualization": {"mode": "selected_only", "method": "print_table"},
            "selected_dataitem": {
                "paradigms": ["all"],
                "modules": ["all"],
                "hyperparameters": ["all"],
                "metrics": ["accuracy"],
            },
            "save_mode": "selected_and_all",
        }
        rank_obj = Rank(config)

        mock_test_case = MagicMock()
        mock_test_case.id = "test_1"
        mock_test_case.output_dir = "/tmp/test_output"

        # Test results where Matrix is None or not present (e.g. single-task learning)
        test_results = {
            "test_1": [{"accuracy": 0.95, "Matrix": None}, 1.23]
        }

        # Should execute cleanly without raising AttributeError
        try:
            rank_obj._draw_pictures([mock_test_case], test_results)
        except AttributeError as err:
            self.fail(f"_draw_pictures raised AttributeError unexpectedly: {err}")


if __name__ == "__main__":
    unittest.main()
