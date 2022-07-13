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

"""Visualization"""

import sys

from prettytable import from_csv


def print_table(rank_file):
    """ print rank of the test"""
    with open(rank_file, "r", encoding="utf-8") as file:
        table = from_csv(file)
        print(table)


def get_visualization_func(mode):
    """ get visualization func """
    return getattr(sys.modules[__name__], mode)
