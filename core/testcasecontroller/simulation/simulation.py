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
    cluster_name : int
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
            if attribute in self.__dict__:
                self.__dict__[attribute] = value

        self._check_fields()

    def _check_fields(self):
        """
        check the fields of simulation config.
        """
        if not isinstance(self.cloud_number, int):
            raise ValueError(f"simulation cloud_number" f"({self.cloud_number} must be int type.")

        if not isinstance(self.edge_number, int):
            raise ValueError(f"simulation edge_number" f"({self.edge_number} must be int type.")

        if not isinstance(self.cluster_name, str):
            raise ValueError(f"simulation ({self.cluster_name}) must be string type.")

        if not isinstance(self.kubeedge_version, str):
            raise ValueError(f"simulation ({self.kubeedge_version}) must be string type.")

        if not isinstance(self.sedna_version, str):
            raise ValueError(f"simulation ({self.sedna_version}) must be string type.")
