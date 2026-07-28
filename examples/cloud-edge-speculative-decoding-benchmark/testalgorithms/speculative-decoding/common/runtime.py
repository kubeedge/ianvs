"""Runtime helpers shared by benchmark wrappers."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import random
import time
from typing import Any

import torch

from common.config import _to_bool, _to_int, _to_optional_int
from common.payload import network_bytes_from_payloads
from common.timing import now


def _resolve_output_log_path(configured_path, default_filename):
    """Resolve the sample-output log path under the Ianvs result directory."""
    result_root = os.environ.get("RESULT_SAVED_URL", ".")
    if configured_path:
        expanded_path = os.path.expanduser(str(configured_path))
        if os.path.isabs(expanded_path):
            return expanded_path
        return os.path.join(result_root, expanded_path)
    return os.path.join(result_root, default_filename)


def record_sample_output(config_source, payload):
    """Append one sample-output record to the configured JSONL log."""
    output_log = _resolve_output_log_path(
        getattr(config_source, "sample_output_log", None),
        "specdec_sample_outputs.jsonl",
    )
    try:
        os.makedirs(os.path.dirname(output_log) or ".", exist_ok=True)
        with open(output_log, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def release_cuda_memory():
    """Best-effort cleanup for Python references and CUDA allocator state."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


class NetworkSimulator:
    """Small network-delay simulator for cloud-edge payload exchanges."""

    def __init__(
        self,
        *,
        enable_sleep=False,
        rtt_ms=0.0,
        jitter_ms=0.0,
        uplink_ratio=0.5,
        uplink_bandwidth_mbps=0.0,
        downlink_bandwidth_mbps=0.0,
        seed=42,
        include_breakdown=False,
    ):
        self.enable_sleep = bool(enable_sleep)
        self.rtt_ms = max(0.0, float(rtt_ms or 0.0))
        self.jitter_ms = max(0.0, float(jitter_ms or 0.0))
        self.uplink_ratio = min(1.0, max(0.0, float(uplink_ratio or 0.5)))
        self.uplink_bandwidth_mbps = max(0.0, float(uplink_bandwidth_mbps or 0.0))
        self.downlink_bandwidth_mbps = max(0.0, float(downlink_bandwidth_mbps or 0.0))
        self.rng = random.Random(int(seed))
        self.include_breakdown = bool(include_breakdown)

    @staticmethod
    def bandwidth_delay_ms(num_bytes, bandwidth_mbps):
        """Estimate link transfer time from bytes and bandwidth."""
        if bandwidth_mbps <= 0.0 or num_bytes <= 0:
            return 0.0
        return (float(num_bytes) * 8.0) / (float(bandwidth_mbps) * 1_000_000.0) * 1000.0

    def sample_base_delays(self):
        """Sample one uplink/downlink propagation split from RTT."""
        jitter = 0.0 if self.jitter_ms <= 0.0 else self.rng.uniform(-self.jitter_ms, self.jitter_ms)
        total = max(self.rtt_ms + jitter, 0.0)
        return total * self.uplink_ratio, total * (1.0 - self.uplink_ratio)

    def simulate(self, uplink_bytes, downlink_bytes):
        """Simulate one payload exchange and return timing metadata."""
        up_base_ms, down_base_ms = self.sample_base_delays()
        uplink_transfer_ms = self.bandwidth_delay_ms(uplink_bytes, self.uplink_bandwidth_mbps)
        downlink_transfer_ms = self.bandwidth_delay_ms(downlink_bytes, self.downlink_bandwidth_mbps)
        uplink_ms = up_base_ms + uplink_transfer_ms
        downlink_ms = down_base_ms + downlink_transfer_ms

        if self.enable_sleep and uplink_ms > 0.0:
            time.sleep(uplink_ms / 1000.0)
        if self.enable_sleep and downlink_ms > 0.0:
            time.sleep(downlink_ms / 1000.0)

        result = {
            "uplink_bytes": int(uplink_bytes),
            "downlink_bytes": int(downlink_bytes),
            "uplink_ms": uplink_ms,
            "downlink_ms": downlink_ms,
            "network_ms": uplink_ms + downlink_ms,
        }
        if self.include_breakdown:
            result.update(
                {
                    "uplink_propagation_ms": up_base_ms,
                    "downlink_propagation_ms": down_base_ms,
                    "uplink_transfer_ms": uplink_transfer_ms,
                    "downlink_transfer_ms": downlink_transfer_ms,
                    "propagation_ms": up_base_ms + down_base_ms,
                    "transfer_ms": uplink_transfer_ms + downlink_transfer_ms,
                }
            )
        return result
_SESSION_STORE = {}
_CONVERSATION_HISTORY = {}


def get(request_id, default=None):
    """Return one shared request state."""
    return _SESSION_STORE.get(str(request_id), default)


def set_value(request_id, value):
    """Store one shared request state."""
    _SESSION_STORE[str(request_id)] = value
    return value


def pop(request_id, default=None):
    """Delete and return one shared request state."""
    return _SESSION_STORE.pop(str(request_id), default)


def clear():
    """Drop all shared request states."""
    _SESSION_STORE.clear()


def get_or_create(request_id, factory):
    """Return one shared request state, creating it lazily when absent."""
    key = str(request_id)
    value = _SESSION_STORE.get(key)
    if value is None:
        value = factory()
        _SESSION_STORE[key] = value
    return value


def normalize_request(data, default_prompt_tokens, default_completion_tokens, to_optional_int):
    """
    Normalize one dataset sample into the request schema used by the runtime.

    The benchmark may hand us a dict, a JSON string, or plain text.
    """
    if isinstance(data, dict):
        payload = dict(data)
    elif isinstance(data, str):
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            payload = {"query": data}
    else:
        payload = {"query": str(data)}

    query_field = payload.get("query", "")
    if isinstance(query_field, str):
        stripped = query_field.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                nested = json.loads(stripped)
                if isinstance(nested, dict):
                    payload = {**payload, **nested}
            except json.JSONDecodeError:
                pass

    text = payload.get("query", "")
    if not isinstance(text, str):
        text = str(text)

    request_id = payload.get("request_id")
    if not request_id:
        request_id = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]

    prompt_tokens = to_optional_int(payload.get("prompt_tokens"), default_prompt_tokens)
    completion_tokens = to_optional_int(
        payload.get("completion_tokens", payload.get("max_new_tokens")),
        default_completion_tokens,
    )

    return {
        "request_id": request_id,
        "query": text,
        "gold": str(payload.get("gold", "")),
        "prompt_tokens": max(1, int(prompt_tokens)) if prompt_tokens is not None else None,
        "completion_tokens": max(1, int(completion_tokens)) if completion_tokens is not None else None,
        "task_name": payload.get("task_name", "default"),
        "dataset_name": payload.get("dataset_name"),
        "stop_mode": payload.get("stop_mode"),
        "sample_index": payload.get("sample_index"),
        "warmup_samples": payload.get("warmup_samples", 0),
        "is_warmup": bool(payload.get("is_warmup", False)),
        "conversation_id": payload.get("conversation_id"),
        "turn_index": payload.get("turn_index"),
        "turn_count": payload.get("turn_count"),
        "question_id": payload.get("question_id"),
        "category": payload.get("category"),
    }


def _conversation_scope(wrapper):
    """Return the history scope used to isolate algorithm execution modes."""
    return (
        getattr(wrapper, "algorithm_name", wrapper.__class__.__name__),
        getattr(wrapper, "inference_mode", getattr(wrapper, "role", "")),
        getattr(wrapper, "model_name", ""),
    )


def build_conversation_messages(wrapper, request):
    """Build chat-template messages for single-turn or MT-Bench turn samples."""
    conversation_id = request.get("conversation_id")
    turn_index = request.get("turn_index")
    query = str(request.get("query", ""))
    if not conversation_id or turn_index is None:
        return [{"role": "user", "content": query}]

    key = (_conversation_scope(wrapper), str(conversation_id))
    history = list(_CONVERSATION_HISTORY.get(key, []))
    current_turn = int(turn_index)
    if current_turn <= 0:
        history = []
    messages = list(history)
    messages.append({"role": "user", "content": query})
    return messages


def build_prompt_text(wrapper, tokenizer, request, *, add_generation_prompt=True):
    """Build model prompt text, using chat templates for multi-turn samples."""
    messages = build_conversation_messages(wrapper, request)
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
    if len(messages) == 1:
        return str(messages[0]["content"])
    rendered = []
    for message in messages:
        role = str(message.get("role", "user")).capitalize()
        rendered.append(f"{role}: {message.get('content', '')}")
    if add_generation_prompt:
        rendered.append("Assistant:")
    return "\n".join(rendered)


def remember_conversation_response(wrapper, request, completion):
    """Store one generated answer so the next MT-Bench turn gets real history."""
    conversation_id = request.get("conversation_id")
    turn_index = request.get("turn_index")
    if not conversation_id or turn_index is None:
        return

    key = (_conversation_scope(wrapper), str(conversation_id))
    history = [] if int(turn_index) <= 0 else list(_CONVERSATION_HISTORY.get(key, []))
    history.append({"role": "user", "content": str(request.get("query", ""))})
    history.append({"role": "assistant", "content": str(completion or "")})
    _CONVERSATION_HISTORY[key] = history


def compute_perf(total_latency_ms, completion_tokens, ttft_ms):
    """Compute TTFT / ITL / throughput from millisecond latency totals."""
    total_latency_ms = max(float(total_latency_ms), 1e-6)
    completion_tokens = max(int(completion_tokens), 1)
    ttft_ms = max(float(ttft_ms), 1e-6)

    if completion_tokens <= 1:
        itl_s = total_latency_ms / 1000.0
    else:
        itl_s = max((total_latency_ms - ttft_ms) / (completion_tokens - 1) / 1000.0, 1e-6)

    throughput = completion_tokens / (total_latency_ms / 1000.0)
    return ttft_ms / 1000.0, itl_s, throughput


def build_single_path_response(
    prompt_tokens,
    completion_tokens,
    completion_text,
    perf,
    simulation,
    timestamps=None,
    benchmark=None,
):
    """Build the single-path response schema expected by Ianvs metrics."""
    response = {
        "completion": completion_text,
        "usage": {
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(prompt_tokens) + int(completion_tokens),
        },
        "perf": {
            "time_to_first_token": round(perf[0], 6),
            "internal_token_latency": round(perf[1], 6),
            "throughput": round(perf[2], 6),
        },
        "simulation": simulation,
    }
    if timestamps is not None:
        response["timestamps"] = timestamps
    if benchmark is not None:
        response["benchmark"] = dict(benchmark)
    return response

def _build_benchmark_payload(request):
    """Extract benchmark-only metadata that should round-trip to metric scripts."""
    if request is None:
        return None

    sample_index = request.get("sample_index")
    warmup_samples = request.get("warmup_samples")
    is_warmup = request.get("is_warmup")
    if sample_index is None and warmup_samples is None and is_warmup is None:
        return None

    return {
        "sample_index": sample_index,
        "warmup_samples": int(warmup_samples or 0),
        "is_warmup": bool(is_warmup),
    }


def compute_specdec_perf(total_latency_ms, completion_tokens, ttft_ms, prefill_ms=None):
    """Return wall-clock and decode-only metrics."""
    total_latency_ms = float(total_latency_ms)
    completion_tokens = int(completion_tokens)
    ttft_ms = float(ttft_ms)
    if prefill_ms is None:
        prefill_ms = ttft_ms
    prefill_ms = float(prefill_ms)
    wall_clock_throughput = 0.0 if total_latency_ms <= 0 else completion_tokens / (total_latency_ms / 1000.0)
    decode_latency_ms = max(total_latency_ms - ttft_ms, 0.0)
    decode_throughput = 0.0 if decode_latency_ms <= 0 else completion_tokens / (decode_latency_ms / 1000.0)
    itl_ms = 0.0 if completion_tokens <= 1 else max(total_latency_ms - ttft_ms, 0.0) / (completion_tokens - 1)
    return {
        "prefill_ms": prefill_ms,
        "ttft_ms": ttft_ms,
        "throughput_toks_per_s": wall_clock_throughput,
        "wall_clock_throughput_toks_per_s": wall_clock_throughput,
        "decode_throughput_toks_per_s": decode_throughput,
        "decode_latency_ms": decode_latency_ms,
        "itl_ms": itl_ms,
        "e2e_latency_ms": total_latency_ms,
    }


def build_perf_payload(perf):
    """Convert runtime metrics into the Ianvs perf schema."""
    return {
        "time_to_first_token": round(float(perf.get("ttft_ms", 0.0)) / 1000.0, 6),
        "internal_token_latency": round(float(perf.get("itl_ms", 0.0)) / 1000.0, 6),
        "throughput": round(float(perf.get("wall_clock_throughput_toks_per_s", perf.get("throughput_toks_per_s", 0.0))), 6),
        "prefill_ms": round(float(perf.get("prefill_ms", 0.0)), 6),
        "ttft_ms": round(float(perf.get("ttft_ms", 0.0)), 6),
        "throughput_toks_per_s": round(float(perf.get("throughput_toks_per_s", 0.0)), 6),
        "wall_clock_throughput_toks_per_s": round(float(perf.get("wall_clock_throughput_toks_per_s", 0.0)), 6),
        "decode_throughput_toks_per_s": round(float(perf.get("decode_throughput_toks_per_s", 0.0)), 6),
        "decode_latency_ms": round(float(perf.get("decode_latency_ms", 0.0)), 6),
        "itl_ms": round(float(perf.get("itl_ms", 0.0)), 6),
        "e2e_latency_ms": round(float(perf.get("e2e_latency_ms", 0.0)), 6),
    }


def build_specdec_response(
    tokenizer,
    request,
    prompt_token_count,
    completion_ids,
    perf,
    simulation,
    *,
    token_provenance=None,
    round_sequence=None,
    extra_fields=None,
):
    """Build one benchmark-facing response object."""
    completion_text = tokenizer.decode(list(completion_ids or []), skip_special_tokens=True)
    response = {
        "request_id": request.get("request_id"),
        "task_name": request.get("task_name", "default"),
        "completion": completion_text,
        "usage": {
            "prompt_tokens": int(prompt_token_count),
            "completion_tokens": int(len(completion_ids or [])),
            "total_tokens": int(prompt_token_count) + int(len(completion_ids or [])),
        },
        "perf": build_perf_payload(perf),
        "simulation": dict(simulation or {}),
    }
    if token_provenance is not None:
        response["token_provenance"] = list(token_provenance)
    if round_sequence is not None:
        response["round_sequence"] = list(round_sequence)
    benchmark = _build_benchmark_payload(request)
    if benchmark is not None:
        response["benchmark"] = benchmark
    if extra_fields:
        response.update(dict(extra_fields))
    return response



def resolve_stop_on_eos(kwargs):
    """Resolve native EOS-stop behavior from Ianvs profile fields."""
    if "stop_on_eos" in kwargs:
        return _to_bool(kwargs.get("stop_on_eos"), True)
    stop_mode = str(kwargs.get("stop_mode", "choice") or "choice").strip().lower()
    return stop_mode not in {"none", "disabled", "off", "false"}


def build_edge_simulation(request, perf, draft_tokens_per_step, stop_reason):
    """Build edge-only simulation metadata."""
    return {
        "mode": "edge-only",
        "routed_to": "edge",
        "acceptance_rate": "",
        "end_to_end_latency": round(float(perf.get("e2e_latency_ms", 0.0)) / 1000.0, 6),
        "rounds": 1,
        "accepted_draft_tokens": 0,
        "corrected_tokens": 0,
        "total_draft_tokens": 0,
        "network_overhead_ms": 0.0,
        "network_rtt_ms": 0.0,
        "network_jitter_ms": 0.0,
        "draft_tokens_per_step": int(draft_tokens_per_step),
        "task_name": request.get("task_name", "default"),
        "stop_reason": stop_reason,
    }


def build_cloud_simulation(
    request,
    perf,
    draft_tokens_per_step,
    stop_reason,
    network_ms,
    network_rtt_ms,
    network_jitter_ms,
):
    """Build cloud-only simulation metadata."""
    return {
        "mode": "cloud-only",
        "routed_to": "cloud",
        "acceptance_rate": "",
        "end_to_end_latency": round(float(perf.get("e2e_latency_ms", 0.0)) / 1000.0, 6),
        "rounds": 1,
        "accepted_draft_tokens": 0,
        "corrected_tokens": 0,
        "total_draft_tokens": 0,
        "network_overhead_ms": round(float(network_ms), 6),
        "network_rtt_ms": round(float(network_rtt_ms), 6),
        "network_jitter_ms": round(float(network_jitter_ms), 6),
        "draft_tokens_per_step": int(draft_tokens_per_step),
        "task_name": request.get("task_name", "default"),
        "stop_reason": stop_reason,
    }


def build_collaboration_simulation(
    shared_state,
    perf,
    draft_tokens_per_step,
    *,
    mode,
):
    """Build collaboration simulation metadata."""
    total_draft_tokens = int(shared_state.get("total_draft_tokens", 0))
    accepted_draft_tokens = int(shared_state.get("accepted_draft_tokens", 0))
    acceptance_rate = (
        float(accepted_draft_tokens) / float(total_draft_tokens)
        if total_draft_tokens > 0
        else None
    )
    return {
        "mode": mode,
        "routed_to": "collaboration",
        "acceptance_rate": round(acceptance_rate, 6) if acceptance_rate is not None else "",
        "end_to_end_latency": round(float(perf.get("e2e_latency_ms", 0.0)) / 1000.0, 6),
        "rounds": int(len(shared_state.get("round_sequence", []))),
        "accepted_draft_tokens": accepted_draft_tokens,
        "corrected_tokens": int(shared_state.get("corrected_tokens", 0)),
        "total_draft_tokens": total_draft_tokens,
        "network_overhead_ms": round(float(shared_state.get("network_ms", 0.0)), 6),
        "network_rtt_ms": round(float(shared_state.get("network_rtt_ms", 0.0)), 6),
        "network_jitter_ms": round(float(shared_state.get("network_jitter_ms", 0.0)), 6),
        "draft_tokens_per_step": int(draft_tokens_per_step),
        "task_name": shared_state["request"].get("task_name", "default"),
        "stop_reason": str(shared_state.get("stop_reason", "") or ""),
    }


def resolve_device(configured_device: str | None) -> str:
    """Resolve `auto` device settings for model wrappers."""
    if (configured_device or "auto") == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(configured_device)


def init_model_runtime(
    wrapper: Any,
    kwargs: dict[str, Any],
    *,
    default_model: str,
    default_draft_tokens_per_step: int,
    allowed_modes: set[str] | None = None,
    default_mode: str = "collaboration",
    export_model_path: bool = False,
):
    """Initialize common model/runtime config fields on a wrapper."""
    wrapper.kwargs = dict(kwargs)
    wrapper.model_name = kwargs.get("model", default_model)
    if export_model_path:
        os.environ["model_path"] = wrapper.model_name
    if allowed_modes is not None:
        wrapper.inference_mode = kwargs.get("inference_mode", default_mode)
        if wrapper.inference_mode not in allowed_modes:
            allowed = "/".join(sorted(allowed_modes))
            raise ValueError(
                f"Unsupported inference_mode: {wrapper.inference_mode}. "
                f"Expected one of {allowed}."
            )
    wrapper.device = resolve_device(kwargs.get("device", "auto"))
    wrapper.trust_remote_code = _to_bool(kwargs.get("trust_remote_code", True), True)
    wrapper.default_prompt_tokens = _to_optional_int(kwargs.get("prompt_tokens"))
    wrapper.default_completion_tokens = _to_optional_int(kwargs.get("max_new_tokens"), 64)
    wrapper.draft_tokens_per_step = max(
        1,
        _to_int(kwargs.get("draft_tokens_per_step"), default_draft_tokens_per_step),
    )
    wrapper.temperature = float(
        kwargs.get("sample_temperature", kwargs.get("temperature", 0.0)) or 0.0
    )
    wrapper.stop_on_eos = resolve_stop_on_eos(kwargs)
    wrapper.sample_output_log = kwargs.get("sample_output_log")
    wrapper.tokenizer = None
    wrapper.model = None


def init_network_runtime(wrapper: Any, kwargs: dict[str, Any], *, include_breakdown: bool = False):
    """Initialize network simulation config fields on a verifier wrapper."""
    wrapper.enable_network_sleep = _to_bool(kwargs.get("enable_network_sleep", False), False)
    wrapper.network_rtt_ms = max(0.0, float(kwargs.get("network_rtt_ms", 0.0) or 0.0))
    wrapper.network_jitter_ms = max(0.0, float(kwargs.get("network_jitter_ms", 0.0) or 0.0))
    wrapper.network_uplink_ratio = min(
        1.0,
        max(0.0, float(kwargs.get("network_uplink_ratio", 0.5) or 0.5)),
    )
    wrapper.network_uplink_bandwidth_mbps = max(
        0.0,
        float(kwargs.get("network_uplink_bandwidth_mbps", 0.0) or 0.0),
    )
    wrapper.network_downlink_bandwidth_mbps = max(
        0.0,
        float(kwargs.get("network_downlink_bandwidth_mbps", 0.0) or 0.0),
    )
    wrapper.network = NetworkSimulator(
        enable_sleep=wrapper.enable_network_sleep,
        rtt_ms=wrapper.network_rtt_ms,
        jitter_ms=wrapper.network_jitter_ms,
        uplink_ratio=wrapper.network_uplink_ratio,
        uplink_bandwidth_mbps=wrapper.network_uplink_bandwidth_mbps,
        downlink_bandwidth_mbps=wrapper.network_downlink_bandwidth_mbps,
        seed=kwargs.get("network_seed", 42),
        include_breakdown=include_breakdown,
    )


def build_request(wrapper: Any, data: Any) -> dict[str, Any]:
    """Normalize one Ianvs dataset sample using wrapper defaults."""
    return normalize_request(
        data,
        wrapper.default_prompt_tokens,
        wrapper.default_completion_tokens,
        _to_optional_int,
    )


def resolve_completion_limit(wrapper: Any, request: dict[str, Any]) -> int:
    """Resolve a request completion budget using wrapper defaults."""
    limit = request.get("completion_tokens")
    if limit is None:
        limit = wrapper.default_completion_tokens
    return max(int(limit or 1), 1)


def cleanup_model_runtime(wrapper: Any, *core_attrs: str):
    """Release wrapper/core references and clear shared request sessions."""
    wrapper.kwargs = {}
    wrapper.model = None
    wrapper.tokenizer = None
    for attr in core_attrs:
        if hasattr(wrapper.core, attr):
            setattr(wrapper.core, attr, None)
    clear()
    release_cuda_memory()


def decode_tokens(wrapper: Any, token_ids, *, skip_special_tokens: bool = True) -> str:
    """Decode token ids when a tokenizer is attached."""
    if getattr(wrapper, "tokenizer", None) is None:
        return str(list(token_ids or []))
    return wrapper.tokenizer.decode(
        list(token_ids or []),
        skip_special_tokens=skip_special_tokens,
    )


def build_shared_state(
    wrapper: Any,
    request: dict[str, Any],
    *,
    completion_limit: int,
    request_start_time: float,
    prompt_token_count: int = 0,
    draft_prefill_ms: float = 0.0,
    verify_prefill_ms: float = 0.0,
    prefill_ms: float | None = None,
    decode_tokenizer: Any = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one shared request-state dictionary for collaboration mode."""
    if prefill_ms is None:
        prefill_ms = float(draft_prefill_ms) + float(verify_prefill_ms)
    state = {
        "request": dict(request),
        "prompt_token_count": int(prompt_token_count),
        "completion_limit": int(completion_limit),
        "request_start_time": float(request_start_time),
        "timing_device": wrapper.device,
        "draft_prefill_ms": float(draft_prefill_ms),
        "verify_prefill_ms": float(verify_prefill_ms),
        "prefill_ms": float(prefill_ms),
        "ttft_ms": None,
        "committed_ids": [],
        "accepted_draft_tokens": 0,
        "corrected_tokens": 0,
        "total_draft_tokens": 0,
        "network_ms": 0.0,
        "network_propagation_ms": 0.0,
        "network_transfer_ms": 0.0,
        "network_rtt_ms": float(wrapper.kwargs.get("network_rtt_ms", 0.0) or 0.0),
        "network_jitter_ms": float(wrapper.kwargs.get("network_jitter_ms", 0.0) or 0.0),
        "edge_compute_ms": 0.0,
        "cloud_compute_ms": 0.0,
        "round_sequence": [],
        "token_provenance": [],
        "stop_reason": "",
        "finalized": False,
        "decode_tokenizer": decode_tokenizer,
    }
    if extra:
        state.update(extra)
    return state


def start_drafter_session(
    wrapper: Any,
    *,
    data: Any = None,
    request: dict[str, Any] | None = None,
    prepare_prompt=None,
    create_core_session=None,
    shared_state_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a drafter-side session and shared collaboration state."""
    if request is None:
        request = build_request(wrapper, data)
    request = dict(request)
    request["completion_tokens"] = resolve_completion_limit(wrapper, request)
    request_start_time = now(wrapper.device)
    prompt_payload = None
    prompt_ids = None
    prompt_token_count = 0
    draft_session = None
    draft_prefill_ms = 0.0
    if prepare_prompt is not None:
        prompt_payload = prepare_prompt(request)
        prompt_ids = prompt_payload["prompt_ids"]
        prompt_token_count = int(prompt_payload["prompt_token_count"])
    if create_core_session is not None:
        draft_session = create_core_session(prompt_payload)
        draft_prefill_ms = float(getattr(draft_session, "prefill_ms", 0.0))

    shared_state = get_or_create(
        request["request_id"],
        lambda: build_shared_state(
            wrapper,
            request,
            prompt_token_count=prompt_token_count,
            completion_limit=request["completion_tokens"],
            request_start_time=request_start_time,
            draft_prefill_ms=draft_prefill_ms,
            decode_tokenizer=wrapper.tokenizer,
            extra=shared_state_extra,
        ),
    )
    session = {
        "request_id": str(request.get("request_id", "default")),
        "request": request,
        "completion_limit": int(request["completion_tokens"]),
        "_shared_state": shared_state,
    }
    if prompt_ids is not None:
        session["prompt_ids"] = prompt_ids
        session["prompt_token_count"] = prompt_token_count
    if draft_session is not None:
        session["_core_session"] = draft_session
    return session


def resolve_verifier_request(
    wrapper: Any,
    *,
    data: Any = None,
    request: dict[str, Any] | None = None,
    draft_session: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve and normalize a verifier-side request."""
    if request is None:
        request = draft_session.get("request") if isinstance(draft_session, dict) else None
    if request is None:
        request = build_request(wrapper, data)
    request = dict(request)
    request["completion_tokens"] = resolve_completion_limit(wrapper, request)
    return request


def get_required_shared_state(request: dict[str, Any]) -> dict[str, Any]:
    """Return the existing collaboration state or fail with a clear error."""
    shared_state = get(request["request_id"])
    if shared_state is None:
        raise RuntimeError(f"Shared drafter state is missing for request {request['request_id']}.")
    return shared_state


def empty_network_result() -> dict[str, Any]:
    """Return a zero-valued network result with stable keys."""
    return {
        "uplink_bytes": 0,
        "downlink_bytes": 0,
        "uplink_ms": 0.0,
        "downlink_ms": 0.0,
        "network_ms": 0.0,
        "uplink_propagation_ms": 0.0,
        "downlink_propagation_ms": 0.0,
        "uplink_transfer_ms": 0.0,
        "downlink_transfer_ms": 0.0,
        "propagation_ms": 0.0,
        "transfer_ms": 0.0,
    }


def simulate_payloads(wrapper: Any, payloads: list[Any]) -> dict[str, Any]:
    """Account and simulate a list of typed transport payloads."""
    uplink_bytes, downlink_bytes = network_bytes_from_payloads(payloads)
    return wrapper.network.simulate(uplink_bytes, downlink_bytes)


def simulate_optional_payloads(wrapper: Any, payloads: list[Any] | None) -> dict[str, Any]:
    """Simulate payloads only when there is actual data to transfer."""
    payloads = list(payloads or [])
    if not payloads:
        return empty_network_result()
    return simulate_payloads(wrapper, payloads)


def record_verifier_prefill(
    wrapper: Any,
    shared_state: dict[str, Any],
    request: dict[str, Any],
    *,
    prompt_token_count: int,
    verify_prefill_ms: float,
    prefill_ms: float | None = None,
    include_draft_prefill: bool = True,
    prefill_payloads: list[Any] | None = None,
    set_ttft_from_prefill: bool = False,
    seed_token_id: int | None = None,
    seed_source: str = "verifier_seed",
    prefill_network_extra: dict[str, Any] | None = None,
    extra_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record verifier prefill timing, optional network, and seed token state."""
    prefill_network = simulate_optional_payloads(wrapper, prefill_payloads)
    if prefill_ms is None:
        prefill_ms = float(verify_prefill_ms)
        if include_draft_prefill:
            prefill_ms += float(shared_state.get("draft_prefill_ms", 0.0))

    shared_state["request"] = request
    shared_state["prompt_token_count"] = int(prompt_token_count)
    shared_state["verify_prefill_ms"] = float(verify_prefill_ms)
    shared_state["prefill_ms"] = float(prefill_ms)
    if set_ttft_from_prefill:
        shared_state["ttft_ms"] = float(prefill_ms) + (
            float(prefill_network["network_ms"]) if wrapper.enable_network_sleep else 0.0
        )
    shared_state["timing_device"] = wrapper.device
    shared_state["decode_tokenizer"] = wrapper.tokenizer
    add_network_to_shared_state(shared_state, prefill_network)

    if prefill_payloads is not None or prefill_network_extra is not None:
        shared_state["prefill_network"] = {
            "network_ms": float(prefill_network["network_ms"]),
            "propagation_ms": float(prefill_network.get("propagation_ms", 0.0)),
            "transfer_ms": float(prefill_network.get("transfer_ms", 0.0)),
            "uplink_bytes": int(prefill_network["uplink_bytes"]),
            "downlink_bytes": int(prefill_network["downlink_bytes"]),
        }
        if prefill_network_extra:
            shared_state["prefill_network"].update(prefill_network_extra)

    if seed_token_id is not None:
        shared_state["committed_ids"] = [int(seed_token_id)]
        shared_state["token_provenance"] = [
            {
                "position": 0,
                "round": 0,
                "token_id": int(seed_token_id),
                "source": seed_source,
            }
        ]
    if extra_state:
        shared_state.update(extra_state)
    return prefill_network


def build_verifier_session(
    request: dict[str, Any],
    *,
    prompt_ids: Any,
    prompt_token_count: int,
    shared_state: dict[str, Any],
    core_session: Any = None,
) -> dict[str, Any]:
    """Build a verifier-side session dictionary."""
    session = {
        "request_id": str(request.get("request_id", "default")),
        "request": request,
        "prompt_ids": prompt_ids,
        "prompt_token_count": int(prompt_token_count),
        "_shared_state": shared_state,
    }
    if core_session is not None:
        session["_core_session"] = core_session
    return session


def merge_network_stats(target: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Add numeric network timing/byte fields into `target`."""
    for key, value in dict(update or {}).items():
        if isinstance(value, (int, float)):
            target[key] = target.get(key, 0) + value
    return target


def add_network_to_shared_state(shared_state: dict[str, Any], network: dict[str, Any]):
    """Accumulate network totals into a collaboration shared state."""
    shared_state["network_ms"] += float(network.get("network_ms", 0.0) or 0.0)
    shared_state["network_propagation_ms"] += float(network.get("propagation_ms", 0.0) or 0.0)
    shared_state["network_transfer_ms"] += float(network.get("transfer_ms", 0.0) or 0.0)


def build_feedback_payload(draft_output: dict[str, Any], verify_payload: dict[str, Any]) -> dict[str, Any]:
    """Build verifier feedback without recursively embedding itself."""
    return {
        "draft_output": dict(draft_output or {}),
        "verify_output": {
            key: value
            for key, value in dict(verify_payload or {}).items()
            if key != "feedback"
        },
    }


def finalize_stop_and_progress(
    *,
    stop: bool,
    stop_reason: str,
    progress: bool,
    shared_state: dict[str, Any],
) -> tuple[bool, str, bool]:
    """Apply the common stalled-round guard and persist stop reason."""
    if not progress:
        stop = True
        stop_reason = "stalled"
    if stop_reason:
        shared_state["stop_reason"] = stop_reason
    return bool(stop), str(stop_reason or ""), bool(progress)


def apply_verify_round(
    shared_state: dict[str, Any],
    *,
    draft_ids: list[int],
    accepted_ids: list[int],
    corrected_ids: list[int],
    rejected_ids: list[int],
    verify_payload: dict[str, Any],
    network: dict[str, Any],
    draft_output: dict[str, Any],
    token_provenance: list[dict[str, Any]] | None = None,
    round_stats: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], bool, str, bool]:
    """Apply one verify result to shared request state.

    Algorithms provide accepted/corrected/rejected token ids and optional
    per-round provenance/stat fields. This helper handles the common state
    mutation, network accounting, stalled-round guard, and feedback payload.
    """
    draft_ids = list(draft_ids or [])
    accepted_ids = list(accepted_ids or [])
    corrected_ids = list(corrected_ids or [])
    rejected_ids = list(rejected_ids or [])
    new_ids = accepted_ids + corrected_ids
    if shared_state.get("ttft_ms") is None and new_ids:
        shared_state["ttft_ms"] = (
            now(shared_state["timing_device"]) - shared_state["request_start_time"]
        ) * 1000.0

    round_index = len(shared_state.get("round_sequence", [])) + 1
    base_position = len(shared_state.get("token_provenance", []))
    if token_provenance is None:
        token_provenance = []
        for token_id in accepted_ids:
            token_provenance.append({"token_id": int(token_id), "source": "draft_accepted"})
        for token_id in corrected_ids:
            token_provenance.append({"token_id": int(token_id), "source": "verifier_corrected"})
    for offset, item in enumerate(list(token_provenance or [])):
        shared_state["token_provenance"].append(
            {
                "position": base_position + offset,
                "round": round_index,
                **dict(item),
            }
        )

    shared_state["committed_ids"].extend(new_ids)
    shared_state["accepted_draft_tokens"] += len(accepted_ids)
    shared_state["corrected_tokens"] += len(corrected_ids)
    shared_state["total_draft_tokens"] += len(draft_ids)
    add_network_to_shared_state(shared_state, network)
    shared_state["cloud_compute_ms"] += float(verify_payload.get("cloud_compute_ms", 0.0) or 0.0)

    if round_stats is None:
        round_stats = {
            "draft_count": len(draft_ids),
            "accepted_length": len(accepted_ids),
            "corrected_count": len(corrected_ids),
            "rejected_draft_count": len(rejected_ids),
            "rejected_draft_ids": list(rejected_ids),
            "accepted_ids": list(accepted_ids),
            "corrected_ids": list(corrected_ids),
            "stop_reason": str(verify_payload.get("stop_reason", "") or ""),
            "cloud_compute_ms": float(verify_payload.get("cloud_compute_ms", 0.0) or 0.0),
            "edge_compute_ms": float(draft_output.get("edge_compute_ms", 0.0) or 0.0),
        }
    else:
        round_stats = dict(round_stats)
    round_stats["round"] = round_index
    round_stats["network_ms"] = float(network.get("network_ms", 0.0) or 0.0)
    round_stats["network_propagation_ms"] = float(network.get("propagation_ms", 0.0) or 0.0)
    round_stats["network_transfer_ms"] = float(network.get("transfer_ms", 0.0) or 0.0)
    round_stats["uplink_bytes"] = int(network.get("uplink_bytes", 0) or 0)
    round_stats["downlink_bytes"] = int(network.get("downlink_bytes", 0) or 0)
    shared_state["round_sequence"].append(round_stats)

    stop, stop_reason, progress = finalize_stop_and_progress(
        stop=bool(verify_payload.get("stop", False)),
        stop_reason=str(verify_payload.get("stop_reason", "") or ""),
        progress=bool(new_ids or draft_ids or verify_payload.get("stop", False)),
        shared_state=shared_state,
    )
    return round_stats, stop, stop_reason, progress


def build_verify_response(
    *,
    accepted_ids: list[int],
    corrected_ids: list[int],
    rejected_ids: list[int],
    verify_payload: dict[str, Any],
    network: dict[str, Any],
    draft_output: dict[str, Any],
    stop: bool,
    stop_reason: str,
    progress: bool,
    round_stats: dict[str, Any],
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the common verifier response payload."""
    payload = {
        "accepted_length": len(list(accepted_ids or [])),
        "accepted_ids": list(accepted_ids or []),
        "corrected_ids": list(corrected_ids or []),
        "rejected_draft_ids": list(rejected_ids or []),
        "cloud_compute_ms": float(verify_payload.get("cloud_compute_ms", 0.0) or 0.0),
        "network_overhead_ms": float(network.get("network_ms", 0.0) or 0.0),
        "stop": bool(stop),
        "stop_reason": stop_reason,
        "progress": bool(progress),
        "round_stats": round_stats,
    }
    if extra_fields:
        payload.update(extra_fields)
    payload["feedback"] = build_feedback_payload(draft_output, payload)
    return payload


def apply_verify_payloads(
    wrapper: Any,
    shared_state: dict[str, Any],
    *,
    transport_payloads: list[Any],
    draft_ids: list[int],
    accepted_ids: list[int],
    corrected_ids: list[int],
    rejected_ids: list[int],
    verify_payload: dict[str, Any],
    draft_output: dict[str, Any],
    token_provenance: list[dict[str, Any]] | None = None,
    round_stats: dict[str, Any] | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply one verify result, including transport accounting, in adapter code."""
    network = simulate_payloads(wrapper, transport_payloads)
    round_stats, stop, stop_reason, progress = apply_verify_round(
        shared_state=shared_state,
        draft_ids=draft_ids,
        accepted_ids=accepted_ids,
        corrected_ids=corrected_ids,
        rejected_ids=rejected_ids,
        verify_payload=verify_payload,
        network=network,
        draft_output=draft_output,
        token_provenance=token_provenance,
        round_stats=round_stats,
    )
    return build_verify_response(
        accepted_ids=accepted_ids,
        corrected_ids=corrected_ids,
        rejected_ids=rejected_ids,
        verify_payload=verify_payload,
        network=network,
        draft_output=draft_output,
        stop=stop,
        stop_reason=stop_reason,
        progress=progress,
        round_stats=round_stats,
        extra_fields=extra_fields,
    )


def build_collaboration_result(
    wrapper: Any,
    shared_state: dict[str, Any],
    *,
    mode: str,
    algorithm: str | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and log the final collaboration response."""
    completion_ids = list(shared_state.get("committed_ids", []))[: int(shared_state["completion_limit"])]
    total_ms = (now(shared_state["timing_device"]) - shared_state["request_start_time"]) * 1000.0
    perf = compute_specdec_perf(
        total_ms,
        len(completion_ids),
        shared_state.get("ttft_ms") or total_ms,
        prefill_ms=shared_state.get("prefill_ms") or shared_state.get("ttft_ms") or total_ms,
    )
    simulation = build_collaboration_simulation(
        shared_state,
        perf,
        wrapper.draft_tokens_per_step,
        mode=mode,
    )
    fields = {
        "mode": "collaboration",
        "prompt": shared_state["request"].get("query", ""),
        "gold": shared_state["request"].get("gold", ""),
        "completion_tokens": len(completion_ids),
        "accepted_draft_tokens": int(shared_state.get("accepted_draft_tokens", 0)),
        "corrected_tokens": int(shared_state.get("corrected_tokens", 0)),
        "total_draft_tokens": int(shared_state.get("total_draft_tokens", 0)),
        "rounds": int(len(shared_state.get("round_sequence", []))),
        "stop_reason": simulation["stop_reason"],
    }
    if algorithm:
        fields["algorithm"] = algorithm
    if extra_fields:
        fields.update(extra_fields)
    response = build_specdec_response(
        shared_state.get("decode_tokenizer") or wrapper.tokenizer,
        shared_state["request"],
        shared_state["prompt_token_count"],
        completion_ids,
        perf,
        simulation,
        token_provenance=shared_state.get("token_provenance", []),
        round_sequence=shared_state.get("round_sequence", []),
        extra_fields=fields,
    )
    remember_conversation_response(
        wrapper,
        shared_state["request"],
        response["completion"],
    )
    sample_record = {
        "request_id": shared_state["request"].get("request_id"),
        "mode": "collaboration",
        "task_name": shared_state["request"].get("task_name", "default"),
        "prompt": shared_state["request"].get("query", ""),
        "gold": shared_state["request"].get("gold", ""),
        "completion": response["completion"],
        "stop_reason": simulation["stop_reason"],
        "accepted_draft_tokens": int(shared_state.get("accepted_draft_tokens", 0)),
        "corrected_tokens": int(shared_state.get("corrected_tokens", 0)),
        "total_draft_tokens": int(shared_state.get("total_draft_tokens", 0)),
        "rounds": int(len(shared_state.get("round_sequence", []))),
        "token_provenance": list(shared_state.get("token_provenance", [])),
        "round_sequence": list(shared_state.get("round_sequence", [])),
    }
    if algorithm:
        sample_record["algorithm"] = algorithm
    record_sample_output(wrapper, sample_record)
    return response


def close_shared_collaboration_session(
    wrapper: Any,
    session: dict[str, Any],
    *,
    build_result,
    pre_close=None,
) -> dict[str, Any] | None:
    """Finalize and remove one collaboration shared session."""
    if pre_close is not None:
        pre_close(session)
    request_id = session["request_id"]
    shared_state = get(request_id)
    if shared_state is None:
        return None
    if not shared_state.get("finalized", False):
        shared_state["finalized"] = True
        result = build_result(shared_state)
        shared_state["result"] = result
    else:
        result = shared_state.get("result")
    pop(request_id, None)
    return result


def build_edge_only_result(
    wrapper: Any,
    request: dict[str, Any],
    prompt_token_count: int,
    generation: dict[str, Any],
    total_ms: float,
    *,
    algorithm: str | None = None,
) -> dict[str, Any]:
    """Build and log an edge-only response."""
    perf = compute_specdec_perf(
        total_ms,
        len(generation["completion_ids"]),
        generation["ttft_ms"],
        prefill_ms=generation["prefill_ms"],
    )
    simulation = build_edge_simulation(
        request,
        perf,
        wrapper.draft_tokens_per_step,
        generation["stop_reason"],
    )
    fields = {
        "mode": "edge-only",
        "prompt": request.get("query", ""),
        "gold": request.get("gold", ""),
        "completion_tokens": len(generation["completion_ids"]),
        "stop_reason": generation["stop_reason"],
    }
    if algorithm:
        fields["algorithm"] = algorithm
    response = build_specdec_response(
        wrapper.tokenizer,
        request,
        prompt_token_count,
        generation["completion_ids"],
        perf,
        simulation,
        extra_fields=fields,
    )
    remember_conversation_response(wrapper, request, response["completion"])
    sample = {
        "request_id": request.get("request_id"),
        "mode": "edge-only",
        "task_name": request.get("task_name", "default"),
        "prompt": request.get("query", ""),
        "gold": request.get("gold", ""),
        "completion": response["completion"],
        "stop_reason": generation["stop_reason"],
    }
    if algorithm:
        sample["algorithm"] = algorithm
    record_sample_output(wrapper, sample)
    return response


def build_cloud_only_result(
    wrapper: Any,
    request: dict[str, Any],
    prompt_token_count: int,
    generation: dict[str, Any],
    total_ms: float,
    network: dict[str, Any],
    *,
    algorithm: str | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and log a cloud-only response."""
    ttft_ms = float(generation["ttft_ms"])
    if wrapper.enable_network_sleep:
        ttft_ms += float(network.get("uplink_ms", 0.0) or 0.0)
    perf = compute_specdec_perf(
        total_ms,
        len(generation["completion_ids"]),
        ttft_ms,
        prefill_ms=generation["prefill_ms"],
    )
    simulation = build_cloud_simulation(
        request,
        perf,
        wrapper.draft_tokens_per_step,
        generation["stop_reason"],
        network.get("network_ms", 0.0),
        wrapper.network_rtt_ms,
        wrapper.network_jitter_ms,
    )
    fields = {
        "mode": "cloud-only",
        "prompt": request.get("query", ""),
        "gold": request.get("gold", ""),
        "completion_tokens": len(generation["completion_ids"]),
        "stop_reason": generation["stop_reason"],
        "network": network,
    }
    if algorithm:
        fields["algorithm"] = algorithm
    if extra_fields:
        fields.update(extra_fields)
    response = build_specdec_response(
        wrapper.tokenizer,
        request,
        prompt_token_count,
        generation["completion_ids"],
        perf,
        simulation,
        extra_fields=fields,
    )
    remember_conversation_response(wrapper, request, response["completion"])
    sample = {
        "request_id": request.get("request_id"),
        "mode": "cloud-only",
        "task_name": request.get("task_name", "default"),
        "prompt": request.get("query", ""),
        "gold": request.get("gold", ""),
        "completion": response["completion"],
        "stop_reason": generation["stop_reason"],
    }
    if algorithm:
        sample["algorithm"] = algorithm
    record_sample_output(wrapper, sample)
    return response


def run_edge_only_generation(
    wrapper: Any,
    data: Any,
    *,
    encode_prompt,
    generate,
    request: dict[str, Any] | None = None,
    algorithm: str | None = None,
) -> dict[str, Any]:
    """Run an edge-only request with common timing/result handling."""
    if request is None:
        request = build_request(wrapper, data)
    prompt_payload = encode_prompt(request)
    completion_limit = resolve_completion_limit(wrapper, request)
    request["completion_tokens"] = completion_limit
    start = now(wrapper.device)
    generation = generate(prompt_payload, completion_limit)
    total_ms = (now(wrapper.device) - start) * 1000.0
    return build_edge_only_result(
        wrapper,
        request,
        prompt_payload["prompt_token_count"],
        generation,
        total_ms,
        algorithm=algorithm,
    )


def run_cloud_only_generation(
    wrapper: Any,
    data: Any,
    *,
    encode_prompt,
    generate,
    prompt_token_count=None,
    algorithm: str | None = None,
    extra_fields_builder=None,
) -> dict[str, Any]:
    """Run a cloud-only request with common network/result handling."""
    request = build_request(wrapper, data)
    request["completion_tokens"] = resolve_completion_limit(wrapper, request)
    prompt_payload = encode_prompt(request)
    prompt_ids = prompt_payload["prompt_ids"]
    if prompt_token_count is None:
        resolved_prompt_token_count = int(prompt_ids.shape[1])
    else:
        resolved_prompt_token_count = int(prompt_token_count(prompt_payload))
    start = now(wrapper.device)
    network = simulate_payloads(wrapper, [prompt_payload["transport_payload"]])
    generation = generate(request, prompt_payload)
    downlink = simulate_payloads(
        wrapper,
        [prompt_payload["completion_payload"](generation["completion_ids"])],
    )
    merge_network_stats(network, downlink)
    total_ms = (now(wrapper.device) - start) * 1000.0
    extra_fields = None
    if extra_fields_builder is not None:
        extra_fields = extra_fields_builder(
            generation=generation,
            total_ms=total_ms,
            network=network,
        )
    return build_cloud_only_result(
        wrapper,
        request,
        resolved_prompt_token_count,
        generation,
        total_ms,
        network,
        algorithm=algorithm,
        extra_fields=extra_fields,
    )
