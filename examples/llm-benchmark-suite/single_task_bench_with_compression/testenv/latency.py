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

__all__ = ["latency"]

@ClassFactory.register(ClassType.GENERAL, alias="latency")
def latency(predicts, targets=None):
    """
    计算平均推理时间
    """
    if isinstance(predicts, dict):
        timings_list = predicts.get("timings", [])
    else:
        # 如果 predicts 不是字典，无法获取 timings
        timings_list = []
    total_time = 0.0
    count = 0
    for timings in timings_list:
        if 'total time' in timings:
            total_time += timings['total time']
            count += 1
    average_latency = total_time / count if count > 0 else 0.0
    return average_latency