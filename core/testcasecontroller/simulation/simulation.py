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

"""Simulation"""

from core.common.log import LOGGER


# limits enforced by the sedna all-in-one installer
MAX_CLOUD_WORKER_NODES = 2
MAX_EDGE_WORKER_NODES = 3
# The sedna all-in-one node image ships keadm v1.9.1, which downloads the
# devices_v1alpha2_* CRD files from the kubeedge release branch; release-1.16
# and later no longer have them (replaced by v1beta1), so the install fails
# with a 404, and the image's Kubernetes 1.21 is too old for recent KubeEdge.
# v1.14.0 is the newest version verified to deploy cleanly end-to-end; bump
# this once the sedna all-in-one image supports newer KubeEdge.
DEFAULT_KUBEEDGE_VERSION = "v1.14.0"


# pylint: disable=too-few-public-methods
class Simulation:
    """
    Simulation: The simulation enviroment, e.g. config of simulation.

    Parameters
    ----------
    cloud_number : int
        number of the cloud worker.
    edge_number : int
        number of the edge nodes.
    cluster_name : string
        name of the simulation cluster.
    kubeedge_version : string
        version of kubeedge, e.g. v1.14.0 (default; v1.16+ fails with the
        sedna all-in-one installer's bundled keadm), latest.
    sedna_version : string
        version of sedna, e.g. 0.4.3, latest.
    """

    def __init__(self, simulation_config):
        self.cloud_number = 0
        self.edge_number = 0
        self.cluster_name = ""
        self.kubeedge_version = DEFAULT_KUBEEDGE_VERSION
        self.sedna_version = ""
        self._parse_config(simulation_config)

    def _parse_config(self, simulation_config):
        """
        parse the simulation config.
        """
        if not isinstance(simulation_config, dict):
            raise ValueError(
                f"simulation config ({simulation_config}) must be a dict.")

        for attribute, value in simulation_config.items():
            if attribute in self.__dict__:
                if isinstance(value, str) and not value.strip():
                    raise ValueError(f"simulation {attribute} must not be empty.")
                self.__dict__[attribute] = value
            else:
                LOGGER.warning("unknown simulation config key ignored: %s", attribute)

        self._check_fields()

    def _check_fields(self):
        """
        check the fields of simulation config.
        """
        for name in ("cloud_number", "edge_number"):
            value = getattr(self, name)
            # bool is a subclass of int, so reject it explicitly
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"simulation {name} ({value}) must be a non-negative int.")

        if self.cloud_number > MAX_CLOUD_WORKER_NODES:
            raise ValueError(
                f"simulation cloud_number ({self.cloud_number}) must be at most "
                f"{MAX_CLOUD_WORKER_NODES}.")
        if self.edge_number > MAX_EDGE_WORKER_NODES:
            raise ValueError(
                f"simulation edge_number ({self.edge_number}) must be at most "
                f"{MAX_EDGE_WORKER_NODES}.")

        for name in ("cluster_name", "kubeedge_version", "sedna_version"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise ValueError(f"simulation {name} ({value}) must be string type.")
