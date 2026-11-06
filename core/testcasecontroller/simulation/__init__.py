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

"""Simulation sandbox: environment-isolated test case execution and
system-metrics profiling, plus the cluster topology configuration
(``Simulation``) inherited from the 2022 implementation.
"""

from core.testcasecontroller.simulation.config import ResourceQuota, SandboxConfig
from core.testcasecontroller.simulation.controller import SimulationController
from core.testcasecontroller.simulation.simulation import (
    SEDNA_MAX_CLOUD_WORKER_NODES,
    SEDNA_MAX_EDGE_NODES,
    Simulation,
)

__all__ = [
    "Simulation", "SandboxConfig", "ResourceQuota", "SimulationController",
    "SEDNA_MAX_CLOUD_WORKER_NODES", "SEDNA_MAX_EDGE_NODES",
]
