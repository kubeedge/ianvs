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

"""Report request throughput in requests per second."""

from sedna.common.class_factory import ClassType, ClassFactory

__all__ = ["throughput"]


@ClassFactory.register(ClassType.GENERAL, alias="throughput")
def throughput(y_true, y_pred):  # pylint: disable=unused-argument
    """Compute throughput from the total elapsed request time."""
    results_list = y_pred.get("results", [])
    total_latency = sum(result["total_time"] for result in results_list)
    if total_latency <= 0:
        return 0.0
    return len(results_list) * 1000 / total_latency
