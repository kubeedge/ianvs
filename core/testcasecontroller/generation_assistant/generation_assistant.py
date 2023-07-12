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

"""
Generation Assistant:
    assist users to generate test cases based on certain rules or constraints,
    e.g., the range of parameters
"""

from itertools import product


def get_full_combinations(name_values_list):
    """
    get full combinations of multiple arrays

    Parameters
    -------
    name_values_list : List
        e.g.: [(name1, [value1, value2]), (name2, [value3, value4])]

    Returns
    -------
    List
        e.g.: [{name1:value1, name2:value3}, {name1:value1, name2:value4},
               {name1:value2, name2:value3}, {name1:value2, name2:value4}]
    """

    name_list = []
    values_list = []
    for name, values in name_values_list:
        name_list.append(name)
        values_list.append(values)

    name_value_dict_list = []
    for combination_value_list in product(*values_list):
        name_value_dict = dict(zip(name_list, combination_value_list))
        name_value_dict_list.append(name_value_dict)

    return name_value_dict_list
