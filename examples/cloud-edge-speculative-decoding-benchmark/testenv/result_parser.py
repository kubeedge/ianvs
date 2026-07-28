from dataclasses import dataclass, field
from typing import Dict, Optional


def _unwrap_response_dict(response):
    # Benchmark adapters sometimes wrap the real response inside tuples/lists.
    # Pull out the first dict-shaped payload so metric code can stay simple.
    if isinstance(response, dict):
        return response

    if isinstance(response, (list, tuple)):
        dict_items = [item for item in response if isinstance(item, dict)]
        if dict_items:
            return dict_items[0]

    return None


def _perf_from_timestamps(response):
    # Prefer timestamps over precomputed perf fields when available so metrics are
    # always derived from the same raw timing source.
    payload = response or {}
    timestamps = payload.get("timestamps") or {}
    request_end_ns = timestamps.get("request_end_ns")
    if request_end_ns is None:
        return None

    request_end_s = max(float(request_end_ns) / 1_000_000_000.0, 0.0)
    first_token_ns = timestamps.get("first_token_ns", request_end_ns)
    ttft_s = max(float(first_token_ns) / 1_000_000_000.0, 0.0)

    usage = payload.get("usage", {}) or {}
    completion_tokens = max(int(usage.get("completion_tokens", 0)), 1)
    if completion_tokens <= 1:
        itl_s = max(request_end_s, 1e-9)
    else:
        itl_s = max((request_end_s - ttft_s) / (completion_tokens - 1), 1e-9)
    throughput = completion_tokens / max(request_end_s, 1e-9)

    return {
        "time_to_first_token": ttft_s,
        "internal_token_latency": itl_s,
        "throughput": throughput,
        "end_to_end_latency": request_end_s,
    }


@dataclass
class SimulationInfo:
    mode: str
    routed_to: str
    acceptance_rate: Optional[float]
    end_to_end_latency: float
    rounds: int
    accepted_draft_tokens: int
    corrected_tokens: int
    total_draft_tokens: int
    network_overhead_ms: float
    network_rtt_ms: float
    network_jitter_ms: float
    draft_tokens_per_step: int
    scenario: str

    @classmethod
    def from_dict(cls, simulation):
        # Keep parsing permissive because older result files may omit some fields.
        payload = simulation or {}
        acceptance_rate = payload.get("acceptance_rate", 0.0)
        if acceptance_rate in ("", None):
            acceptance_rate = None
        else:
            acceptance_rate = float(acceptance_rate)
        return cls(
            payload.get("mode", ""),
            payload.get("routed_to", ""),
            acceptance_rate,
            float(payload.get("end_to_end_latency", 0.0)),
            int(payload.get("rounds", 0)),
            int(payload.get("accepted_draft_tokens", 0)),
            int(payload.get("corrected_tokens", 0)),
            int(payload.get("total_draft_tokens", 0)),
            float(payload.get("network_overhead_ms", 0.0)),
            float(payload.get("network_rtt_ms", 0.0)),
            float(payload.get("network_jitter_ms", 0.0)),
            int(payload.get("draft_tokens_per_step", 0)),
            payload.get("scenario", ""),
        )


@dataclass
class BenchmarkInfo:
    sample_index: Optional[int]
    warmup_samples: int
    is_warmup: bool

    @classmethod
    def from_dict(cls, benchmark):
        payload = benchmark or {}
        sample_index = payload.get("sample_index")
        if sample_index in ("", None):
            sample_index = None
        else:
            sample_index = int(sample_index)
        return cls(
            sample_index=sample_index,
            warmup_samples=max(int(payload.get("warmup_samples", 0) or 0), 0),
            is_warmup=bool(payload.get("is_warmup", False)),
        )


@dataclass
class Response:
    completion: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    time_to_first_token: float
    internal_token_latency: float
    throughput: float
    simulation: SimulationInfo
    benchmark: BenchmarkInfo
    timestamps: Dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, response):
        response = _unwrap_response_dict(response)
        if response:
            # Re-derive perf from timestamps when possible so parser-side metrics stay
            # consistent even if the response carried stale or rounded perf values.
            derived_perf = _perf_from_timestamps(response)
            perf = response.get("perf", {}) or {}
            simulation = SimulationInfo.from_dict(response.get("simulation"))
            if derived_perf is not None:
                simulation.end_to_end_latency = derived_perf["end_to_end_latency"]
            return cls(
                response.get("completion", ""),
                int(response.get("usage", {}).get("prompt_tokens", 0)),
                int(response.get("usage", {}).get("completion_tokens", 0)),
                int(response.get("usage", {}).get("total_tokens", 0)),
                float(
                    derived_perf["time_to_first_token"]
                    if derived_perf is not None
                    else perf.get("time_to_first_token", 0.0)
                ),
                float(
                    derived_perf["internal_token_latency"]
                    if derived_perf is not None
                    else perf.get("internal_token_latency", 0.0)
                ),
                float(
                    derived_perf["throughput"]
                    if derived_perf is not None
                    else perf.get("throughput", 0.0)
                ),
                simulation,
                BenchmarkInfo.from_dict(response.get("benchmark")),
                dict(response.get("timestamps", {}) or {}),
            )

        return cls(
            "",
            0,
            0,
            0,
            0.0,
            0.0,
            0.0,
            SimulationInfo.from_dict({}),
            BenchmarkInfo.from_dict({}),
            {},
        )


@dataclass
class JointInferenceResult:
    is_hard_example: bool
    result: Response
    edge_result: Response
    cloud_result: Response

    @classmethod
    def from_list(cls, is_hard_example, result, edge_result, cloud_result):
        # Joint inference results may arrive as a 4-tuple or as nested lists.
        # Normalize that shape once here so downstream metrics read one schema.
        if isinstance(result, (list, tuple)):
            dict_items = [item for item in result if isinstance(item, dict)]
            if dict_items:
                result = dict_items[0]
                if edge_result is None and len(dict_items) > 1:
                    edge_result = dict_items[1]
                if cloud_result is None and len(dict_items) > 2:
                    cloud_result = dict_items[2]

        return cls(
            is_hard_example,
            Response.from_dict(result),
            Response.from_dict(edge_result),
            Response.from_dict(cloud_result),
        )


def parse_joint_inference_result(pred):
    # Public parser entry used by metric scripts.
    # It accepts the response shapes produced by this benchmark workflow.
    if isinstance(pred, (list, tuple)):
        if len(pred) >= 4:
            return JointInferenceResult.from_list(pred[0], pred[1], pred[2], pred[3])
        if len(pred) == 1 and isinstance(pred[0], dict):
            pred = pred[0]
        else:
            return JointInferenceResult.from_list(False, pred, None, None)

    if isinstance(pred, dict):
        simulation = pred.get("simulation", {}) or {}
        routed_to = simulation.get("routed_to")
        edge_result = pred if routed_to == "edge" else None
        cloud_result = pred if routed_to == "cloud" else None
        return JointInferenceResult.from_list(False, pred, edge_result, cloud_result)

    return JointInferenceResult.from_list(False, pred, None, None)


def parse_effective_joint_inference_results(y_pred):
    """Parse benchmark results and drop warmup samples before aggregation."""
    infer_res = [parse_joint_inference_result(pred) for pred in y_pred]
    return [item for item in infer_res if not item.result.benchmark.is_warmup]
