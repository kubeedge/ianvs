from dataclasses import dataclass
from typing import TypedDict

@dataclass
class Response:
    completion: str
    prompt_tokens : int
    completion_tokens : int
    total_tokens : int
    time_to_first_token: float
    internal_token_latency: float
    throughput: float

    @classmethod
    def from_dict(cls, response):
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
    is_hard_example : bool
    result : Response
    edge_result: Response
    cloud_result: Response

    @classmethod
    def from_list(cls, is_hard_example, result, edge_result, cloud_reslut):
        return cls(
            is_hard_example,
            Response.from_dict(result),
            Response.from_dict(edge_result),
            Response.from_dict(cloud_reslut),
        )
