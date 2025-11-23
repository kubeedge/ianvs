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

"""Visualization"""

import sys
import os
import matplotlib.pyplot as plt
from prettytable import from_csv


def print_table(rank_file):
    """ print rank of the test"""
    with open(rank_file, "r", encoding="utf-8") as file:
        table = from_csv(file, delimiter=",")
        print(table)

def draw_heatmap_picture(output, title, matrix):
    """
    draw heatmap for results
    """
    plt.figure(figsize=(10, 8), dpi=80)
    plt.imshow(matrix, cmap='bwr', extent=(0.5, len(matrix)+0.5, 0.5, len(matrix)+0.5),
                    origin='lower')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('task round', fontsize=15)
    plt.ylabel('task', fontsize=15)
    plt.title(title, fontsize=15)
    plt.colorbar(format='%.2f')
    output_dir = os.path.join(output, f"output/{title}-heatmap.png")
    plt.savefig(output_dir)
    plt.show()

def get_visualization_func(mode):
    """ get visualization func """
    return getattr(sys.modules[__name__], mode)
=======
version https://git-lfs.github.com/spec/v1
oid sha256:63ea2a3f843f085cc9b3230e011838b4fcb959387a5741e980a2139fd35c8a95
size 1618
>>>>>>> 9676c3e (ya toh aar ya toh par)
