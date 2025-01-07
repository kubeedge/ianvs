# Copyright 2024 The KubeEdge Authors.
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

from dataclasses import dataclass
from typing import TypedDict

@dataclass
class Response:
    """Formatted Response Parser"""

    completion: str
    prompt_tokens : int
    completion_tokens : int
    total_tokens : int
    time_to_first_token: float
    internal_token_latency: float
    throughput: float

    @classmethod
    def from_dict(cls, response):
        """Create a Response object from a dictionary

        Parameters
        ----------
        response : dict
            Formatted Response, See `BaseLLM._format_response()` for more details.

        Returns
        -------
        Response
            `Response` Object
        """

        if response:
            return cls(
                response["completion"],
                response["usage"]["prompt_tokens"],
                response["usage"]["completion_tokens"],
                response["usage"]["total_tokens"],
                response["perf"]["time_to_first_token"],
                response["perf"]["internal_token_latency"],
                response["perf"]["throughput"]
            )
        else:
            return cls("", 0, 0, 0, 0, 0, 0)

@dataclass
class JointInferenceResult:
    """Joint Inference Result Parser"""
    is_hard_example : bool
    result : Response
    edge_result: Response
    cloud_result: Response

    @classmethod
    def from_list(cls, is_hard_example, result, edge_result, cloud_reslut):
        """Create a JointInferenceResult object from a list

        Parameters
        ----------
        is_hard_example : bool
            Whter the example is hard or not
        result : dict
            Formatted Response. See `BaseLLM._format_response()` for more details.
        edge_result : dict
            Formatted Response from the Edge Model. See `BaseLLM._format_response()` for more details.
        cloud_reslut : dict
            Formatted Response from the Cloud Model. See `BaseLLM._format_response()` for more details.

        Returns
        -------
        _type_
            _description_
        """

        return cls(
            is_hard_example,
            Response.from_dict(result),
            Response.from_dict(edge_result),
            Response.from_dict(cloud_reslut),
        )
