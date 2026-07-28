"""Payload collection and byte accounting for speculative decoding."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from common.schema import ControlPayload, CustomPayload, TensorPayload, TokenIds, TransportPayload


TOKEN_ID_BYTES = 4
CONTROL_BYTES = 8


def _sequence_length(value: Any) -> int:
    """Return a conservative logical item count for token-id-like values."""
    if value is None:
        return 0
    if hasattr(value, "numel"):
        return int(value.numel())
    if isinstance(value, (str, bytes, bytearray)):
        return 1
    if isinstance(value, dict):
        return sum(_sequence_length(item) for item in value.values())
    if isinstance(value, Iterable):
        total = 0
        for item in value:
            if isinstance(item, Iterable) and not isinstance(item, (str, bytes, bytearray)):
                total += _sequence_length(item)
            else:
                total += 1
        return total
    return 1


def _tensor_nbytes(value: Any) -> int:
    """Return byte size for tensor-like values."""
    if value is None:
        return 0
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        return int(value.numel()) * int(value.element_size())
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    raise TypeError(f"Cannot infer tensor byte size for payload value: {type(value)!r}")


def estimate_payload_bytes(payload: TransportPayload) -> int:
    """Estimate transported bytes for one typed payload."""
    kind = payload.kind
    value = payload.value

    if kind == "token_ids":
        return _sequence_length(value) * TOKEN_ID_BYTES
    if kind in {"logits", "probs", "hidden_states", "tensor"}:
        return _tensor_nbytes(value)
    if kind == "topk_probs":
        token_ids = value["token_ids"]
        token_probs = value["token_probs"]
        return _tensor_nbytes(token_ids) + _tensor_nbytes(token_probs)
    if kind == "control":
        return CONTROL_BYTES
    if kind == "custom":
        return int(value or 0)
    raise ValueError(f"Unsupported payload kind: {kind}")


def collect_transport_payloads(value: Any) -> list[TransportPayload]:
    """Recursively collect typed transport payloads from an output object."""
    payloads: list[TransportPayload] = []

    def visit(item: Any):
        if item is None:
            return
        if isinstance(item, TransportPayload):
            payloads.append(item)
            return
        if hasattr(item, "data"):
            visit(getattr(item, "data"))
            return
        if isinstance(item, dict):
            for nested in item.values():
                visit(nested)
            return
        if isinstance(item, (list, tuple, set)):
            for nested in item:
                visit(nested)

    visit(value)
    return payloads


def payload_bytes_by_direction(payloads: list[TransportPayload]) -> dict[str, int]:
    """Aggregate payload bytes into edge-to-cloud and cloud-to-edge totals."""
    totals = {
        "edge_to_cloud": 0,
        "cloud_to_edge": 0,
    }
    for payload in payloads:
        totals[payload.direction] += estimate_payload_bytes(payload)
    return totals


def token_payload(value: Any, direction: str, name: str = "token_ids") -> TransportPayload:
    """Build a token-id transport payload."""
    return TransportPayload(
        value=value,
        direction=direction,
        kind="token_ids",
        name=name,
    )


def tensor_payload(value: Any, direction: str, kind: str, name: str = "tensor") -> TransportPayload:
    """Build a tensor-like transport payload."""
    return TransportPayload(
        value=value,
        direction=direction,
        kind=kind,
        name=name,
    )


def control_payload(direction: str = "cloud_to_edge", name: str = "control") -> TransportPayload:
    """Build a fixed-size control payload."""
    return TransportPayload(
        value=None,
        direction=direction,
        kind="control",
        name=name,
    )


def custom_payload(nbytes: int, direction: str, name: str = "custom") -> TransportPayload:
    """Build a payload with an explicit byte size."""
    return TransportPayload(
        value=int(nbytes),
        direction=direction,
        kind="custom",
        name=name,
    )


def draft_distribution_payloads(draft_logits: list[Any], direction: str = "edge_to_cloud") -> list[TransportPayload]:
    """Convert legacy draft distribution objects into typed payloads."""
    payloads: list[TransportPayload] = []
    for index, item in enumerate(list(draft_logits or [])):
        if isinstance(item, dict) and item.get("representation") == "topk_probs":
            payloads.append(
                TransportPayload(
                    value=item,
                    direction=direction,
                    kind="topk_probs",
                    name=f"draft_topk_probs_{index}",
                )
            )
        else:
            payloads.append(
                TransportPayload(
                    value=item,
                    direction=direction,
                    kind="logits",
                    name=f"draft_logits_{index}",
                )
            )
    return payloads


def network_bytes_from_payloads(payloads: list[TransportPayload]) -> tuple[int, int]:
    """Return `(uplink_bytes, downlink_bytes)` from typed payloads."""
    totals = payload_bytes_by_direction(payloads)
    return totals["edge_to_cloud"], totals["cloud_to_edge"]
