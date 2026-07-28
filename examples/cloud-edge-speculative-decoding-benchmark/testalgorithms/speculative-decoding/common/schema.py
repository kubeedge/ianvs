"""Unified speculative-decoding runtime schemas.

These dataclasses describe the data exchanged between algorithm hooks and the
benchmark runtime. They are intentionally algorithm-neutral: AR, block, tree, or
other speculative decoding methods should all report their per-round logical
outputs through the same structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Direction = Literal["edge_to_cloud", "cloud_to_edge"]
PayloadKind = Literal[
    "token_ids",
    "logits",
    "probs",
    "topk_probs",
    "hidden_states",
    "tensor",
    "control",
    "custom",
]


@dataclass
class SpecDecRequest:
    """Normalized request consumed by a speculative decoding runtime."""

    request_id: str
    query: str
    gold: str = ""
    task_name: str = "default"
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    dataset_name: str | None = None
    stop_mode: str | None = None
    sample_index: int | None = None
    warmup_samples: int = 0
    is_warmup: bool = False


@dataclass
class TransportPayload:
    """One logical payload crossing the edge-cloud boundary."""

    value: Any
    direction: Direction
    kind: PayloadKind
    name: str = ""


@dataclass
class TokenIds(TransportPayload):
    """Token-id payload with standard int32 accounting."""

    def __init__(
        self,
        value: Any,
        direction: Direction,
        name: str = "token_ids",
    ):
        super().__init__(
            value=value,
            direction=direction,
            kind="token_ids",
            name=name,
        )


@dataclass
class TensorPayload(TransportPayload):
    """Tensor-like payload accounted by numel times element size."""

    def __init__(
        self,
        value: Any,
        direction: Direction,
        kind: PayloadKind = "tensor",
        name: str = "tensor",
    ):
        super().__init__(
            value=value,
            direction=direction,
            kind=kind,
            name=name,
        )


@dataclass
class ControlPayload(TransportPayload):
    """Small control-message payload."""

    def __init__(
        self,
        value: Any = None,
        direction: Direction = "cloud_to_edge",
        name: str = "control",
    ):
        super().__init__(
            value=value,
            direction=direction,
            kind="control",
            name=name,
        )


@dataclass
class CustomPayload(TransportPayload):
    """Payload with caller-provided byte size stored in `value`."""

    def __init__(
        self,
        nbytes: int,
        direction: Direction,
        name: str = "custom",
    ):
        super().__init__(
            value=int(nbytes),
            direction=direction,
            kind="custom",
            name=name,
        )


@dataclass
class DraftResult:
    """Standard drafter return object consumed by the shared runtime.

    `data` is copied into the final drafter payload unchanged, so algorithms can
    pass arbitrary verifier-side state without the runtime interpreting it.
    """

    draft_ids: list[int] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    edge_compute_ms: float = 0.0
    stop: bool = False
    stop_reason: str = ""

@dataclass
class VerifyResult:
    """Standard verifier return object consumed by the shared runtime.

    `payloads` declares logical edge-cloud transfers. `data` is copied into the
    verifier payload unchanged, allowing algorithms to return custom metadata
    without losing it.
    """

    accepted_ids: list[int] = field(default_factory=list)
    corrected_ids: list[int] = field(default_factory=list)
    rejected_draft_ids: list[int] = field(default_factory=list)
    payloads: list[TransportPayload] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    cloud_compute_ms: float = 0.0
    stop: bool = False
    stop_reason: str = ""
    token_provenance: list[dict[str, Any]] = field(default_factory=list)
    round_stats: dict[str, Any] = field(default_factory=dict)
    extra_fields: dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestState:
    """Runtime-owned mutable state for one request."""

    request: SpecDecRequest
    completion_limit: int
    prompt_ids: Any = None
    prompt_token_count: int = 0
    committed_ids: list[int] = field(default_factory=list)
    round_index: int = 0
    stop: bool = False
    stop_reason: str = ""
    accepted_draft_tokens: int = 0
    corrected_tokens: int = 0
    total_draft_tokens: int = 0
    round_sequence: list[dict[str, Any]] = field(default_factory=list)
    token_provenance: list[dict[str, Any]] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    network: dict[str, float | int] = field(default_factory=dict)
    algorithm_state: dict[str, Any] = field(default_factory=dict)
