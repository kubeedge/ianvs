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

__all__ = ["throughput"]

@ClassFactory.register(ClassType.GENERAL, alias="throughput")
def throughput(y_true, y_pred):
    results_list = y_pred.get('results', [])
    
    total_time = 0.0  # /ms
    num_requests = 0
    for result in results_list:
        if isinstance(result, dict) and 'total_time' in result:
            total_time += result['total_time']
            num_requests += 1
    if total_time > 0:
        throughput_value = num_requests / (total_time / 1000)
    else:
        throughput_value = 0.0
    print(f"Throughput: {throughput_value} requests/second")
    return throughput_value