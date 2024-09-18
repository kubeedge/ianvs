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
    num_requests = len(y_pred)
    fixed_time = 1  

    fixed_throughput = num_requests / fixed_time  # 单位请求/秒
    print(f"Throughput: {fixed_throughput} requests/second")
    return fixed_throughput