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

"""Simulation cluster topology configuration.

Hardened against the defects verified in ``verify_legacy_simulation.py``:
B2 (booleans silently passed the int check), B3 (unknown keys were dropped
instead of rejected), B4 (an empty cluster_name produced a malformed
``CLUSTER_NAME=`` shell argument) and B12 (node counts were never checked
against the Sedna all-in-one backend's own hard ceiling, so a value that
passed every Ianvs-side check still failed deep inside a piped shell
script). Attribute names and ``Simulation.__name__`` are unchanged from the
2022 implementation, since ``BenchmarkingJob`` dispatches on
``str.lower(Simulation.__name__)``.
"""

import re

# The Sedna all-in-one installer hard-codes these ceilings and aborts above
# them (see docs/proposals/simulation/sandbox-engine/ianvs-simulation-sandbox.md
# section 2.3, defect B12). Ianvs now rejects larger values at config-parse
# time instead of letting them fail inside the provisioning script.
SEDNA_MAX_CLOUD_WORKER_NODES = 2
SEDNA_MAX_EDGE_NODES = 3

_FIELDS = (
    "cloud_number", "edge_number", "cluster_name",
    "kubeedge_version", "sedna_version",
)

# Kubernetes-style RFC 1123 DNS label: lowercase alphanumerics and '-',
# starting and ending with an alphanumeric character.
_RFC1123_LABEL = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")


# pylint: disable=too-few-public-methods
class Simulation:
    """
    Simulation: the simulation environment, e.g. config of simulation.

    Parameters
    ----------
    cloud_number : int
        number of the cloud worker.
    edge_number : int
        number of the edge nodes.
    cluster_name : string
        name of the simulation cluster.
    kubeedge_version : string
        version of kubeedge, e.g. 1.8.0, latest.
    sedna_version : string
        version of sedna, e.g. 0.4.3, latest.
    """

    def __init__(self, simulation_config):
        self.cloud_number = 0
        self.edge_number = 0
        self.cluster_name = ""
        self.kubeedge_version = ""
        self.sedna_version = ""
        self._parse_config(simulation_config)

    def _parse_config(self, simulation_config):
        """
        parse the simulation config.
        """
        for attribute, value in simulation_config.items():
            if attribute not in _FIELDS:
                raise ValueError(
                    f"simulation config has unknown field({attribute}); "
                    f"expected one of {_FIELDS}.")
            self.__dict__[attribute] = value

        # The all-in-one installer only resolves the newest release when the
        # variable is empty; the literal string 'latest' 404s on the release
        # asset, so normalise it here rather than passing it through.
        if self.kubeedge_version == "latest":
            self.kubeedge_version = ""

        self._check_fields()

    def _check_fields(self):
        """
        check the fields of simulation config.
        """
        for field in ("cloud_number", "edge_number"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"simulation {field}({value}) must be int type.")

        for field in ("cluster_name", "kubeedge_version", "sedna_version"):
            value = getattr(self, field)
            if not isinstance(value, str):
                raise ValueError(
                    f"simulation {field}({value}) must be string type.")

        if not self.cluster_name:
            raise ValueError(
                "simulation cluster_name must be a non-empty string.")

        if not _RFC1123_LABEL.match(self.cluster_name):
            raise ValueError(
                f"simulation cluster_name({self.cluster_name}) must be a "
                "valid RFC 1123 label: lowercase alphanumeric characters or "
                "'-', starting and ending with an alphanumeric character.")

        if not 1 <= self.cloud_number <= SEDNA_MAX_CLOUD_WORKER_NODES:
            raise ValueError(
                f"simulation cloud_number({self.cloud_number}) is out of "
                f"range: the Sedna all-in-one backend supports 1-"
                f"{SEDNA_MAX_CLOUD_WORKER_NODES} cloud worker nodes.")

        if not 1 <= self.edge_number <= SEDNA_MAX_EDGE_NODES:
            raise ValueError(
                f"simulation edge_number({self.edge_number}) is out of "
                f"range: the Sedna all-in-one backend supports 1-"
                f"{SEDNA_MAX_EDGE_NODES} edge nodes.")
