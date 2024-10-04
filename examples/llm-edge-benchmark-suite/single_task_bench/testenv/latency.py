# Copyright 2023 The KubeEdge Authors.
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

from sedna.common.class_factory import ClassType, ClassFactory
import statistics

__all__ = ["latency"]


@ClassFactory.register(ClassType.GENERAL, alias="latency")
def latency(y_true, y_pred):
    results_list = y_pred.get('results', [])
    num_requests = len(results_list)
    total_latency = 0.0
    for result in results_list:
        # print(result)
        total_latency += result['total_time']
    average_latency = total_latency / num_requests
    return average_latency