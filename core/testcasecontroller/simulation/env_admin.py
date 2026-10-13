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

"""Environment Administrator: the 5 responsibilities from the original
Simulation Controller design (issue #348, PR #35):

1. Parse the system config to establish the simulation
2. Check the host environment
3. Build the simulation environment
4. Deploy the modules needed in the simulation environment
5. Close and delete the simulation environment

Responsibility 5 is the one the 2022 implementation never wired up (B7):
``destory_simulation_enviroment()`` existed and was re-exported, but had no
call site anywhere in ``core/``, so every run leaked a cluster. Here it is
owned by ``destroy()``, which every caller runs from a ``finally`` block.
"""

from core.common.constant import SandboxMode
from core.testcasecontroller.simulation import hostcheck


class SimulationEnvironmentAdministrator:
    """Resolves the sandbox tier and owns the cluster's lifecycle, if any."""

    def __init__(self, sandbox_config, simulation=None):
        self.sandbox_config = sandbox_config
        self.simulation = simulation
        self._cluster_backend = None

    def resolve_mode(self):
        """
        Decide which tier this run actually uses.

        ``auto`` degrades to the process tier when a cluster cannot be
        built. ``cluster`` never degrades silently: if the config or host
        can't support it, that is an error, because a silent downgrade
        would report cluster-topology numbers that were never measured on
        one.
        """
        mode = self.sandbox_config.mode

        if mode == SandboxMode.PROCESS.value:
            return SandboxMode.PROCESS.value

        if mode == SandboxMode.CLUSTER.value:
            if self.simulation is None:
                raise ValueError(
                    "sandbox mode is 'cluster' but the benchmarkingjob has "
                    "no 'simulation' block to describe the topology.")
            return SandboxMode.CLUSTER.value

        # mode == "auto"
        if self.simulation is None or not hostcheck.cluster_tier_ready():
            return SandboxMode.PROCESS.value
        return SandboxMode.CLUSTER.value

    def build(self):
        """Provision the cluster tier, if the resolved mode needs one."""
        if self.resolve_mode() != SandboxMode.CLUSTER.value:
            return None
        # Imported lazily: the cluster backend pulls in kubeutil/job_admin,
        # which process-tier-only runs (the common case) never need to load.
        # pylint: disable=import-outside-toplevel
        from core.testcasecontroller.simulation.sandbox.cluster import ClusterBackend
        self._cluster_backend = ClusterBackend(self.simulation)
        self._cluster_backend.build()
        return self._cluster_backend

    def destroy(self):
        """
        Tear down anything ``build()`` provisioned.

        Always safe to call, including before ``build()`` ever ran and more
        than once -- callers run this from a ``finally`` block so it
        survives an exception or Ctrl-C.
        """
        if self._cluster_backend is None:
            return
        try:
            self._cluster_backend.destroy()
        finally:
            self._cluster_backend = None
