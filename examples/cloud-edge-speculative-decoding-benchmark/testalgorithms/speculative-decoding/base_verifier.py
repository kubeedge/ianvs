"""Reusable Ianvs verifier module base for speculative decoding algorithms."""

from __future__ import annotations

from typing import Any

import torch

from common.runtime import apply_verify_payloads
from common.runtime import build_feedback_payload
from common.runtime import build_verifier_session
from common.runtime import cleanup_model_runtime
from common.runtime import get_required_shared_state
from common.runtime import init_model_runtime
from common.runtime import init_network_runtime
from common.runtime import record_verifier_prefill
from common.runtime import resolve_verifier_request
from common.runtime import run_cloud_only_generation
from common.schema import VerifyResult


class BaseSpeculativeVerifier:
    """Common Ianvs-facing verifier lifecycle.

    Subclasses provide algorithm-specific hooks for model loading, prompt
    encoding, verify-state initialization, one verify step, payload declaration,
    and optional cloud-only generation. The base class owns request resolution,
    network simulation, shared-state updates, round metrics, feedback payloads,
    cloud-only result formatting, and cleanup.
    """

    algorithm_name = "base"
    role = "verifier"
    default_model = ""
    default_draft_tokens_per_step = 8
    include_network_breakdown = False
    supports_cloud_only = True
    cloud_only_algorithm: str | None = None
    cleanup_core_attrs = ("model", "tokenizer")

    def __init__(self, **kwargs):
        init_model_runtime(
            self,
            kwargs,
            default_model=self.default_model,
            default_draft_tokens_per_step=self.default_draft_tokens_per_step,
        )
        init_network_runtime(
            self,
            kwargs,
            include_breakdown=self.include_network_breakdown,
        )
        self.core = self.build_core()

    def build_core(self):
        """Build the algorithm-specific verifier implementation."""
        raise NotImplementedError

    def load_core(self):
        """Load algorithm-specific verifier resources."""
        self.core.load()
        self.tokenizer = getattr(self.core, "tokenizer", None)
        self.model = getattr(self.core, "model", None)

    def load(self, *args, **kwargs):
        """Load the verifier model."""
        del args, kwargs
        self.load_core()

    def cleanup(self):
        """Release loaded resources and request sessions."""
        cleanup_model_runtime(self, *self.cleanup_core_attrs)
        self.after_cleanup()

    def after_cleanup(self):
        """Optional subclass hook after common resource cleanup."""

    def resolve_prompt_ids(
        self,
        request: dict[str, Any],
        draft_session: dict[str, Any] | None = None,
    ):
        """Return verifier prompt ids for a normalized request."""
        del draft_session
        return self.encode_prompt(request)

    def encode_prompt(self, request: dict[str, Any]):
        """Return device-local prompt ids for a normalized request."""
        raise NotImplementedError

    def init_verify_state(
        self,
        request: dict[str, Any],
        prompt_ids,
        draft_session: dict[str, Any] | None = None,
        shared_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Initialize verifier-side state and prefill metadata."""
        del request, draft_session, shared_state
        verify_session = self.core.start_session(prompt_ids)
        return {
            "core_session": verify_session,
            "verify_prefill_ms": float(getattr(verify_session, "prefill_ms", 0.0)),
        }

    def verifier_prefill_options(
        self,
        init_payload: dict[str, Any],
        request: dict[str, Any],
        prompt_ids,
        draft_session: dict[str, Any] | None = None,
        shared_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return options passed to common prefill recording."""
        del init_payload, request, prompt_ids, draft_session, shared_state
        return {}

    def after_start_session(
        self,
        session: dict[str, Any],
        init_payload: dict[str, Any],
        draft_session: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Optional subclass hook after common verifier session creation."""
        del init_payload, draft_session
        return session

    def start_session(self, data=None, request=None, draft_session=None, **kwargs):
        """Create one verify-side collaboration session."""
        del kwargs
        request = resolve_verifier_request(
            self,
            data=data,
            request=request,
            draft_session=draft_session,
        )
        prompt_ids = self.resolve_prompt_ids(request, draft_session=draft_session)
        prompt_token_count = int(prompt_ids.shape[1])
        shared_state = get_required_shared_state(request)
        init_payload = self.init_verify_state(
            request,
            prompt_ids,
            draft_session=draft_session,
            shared_state=shared_state,
        )
        record_options = self.verifier_prefill_options(
            init_payload,
            request,
            prompt_ids,
            draft_session=draft_session,
            shared_state=shared_state,
        )
        record_verifier_prefill(
            self,
            shared_state,
            request,
            prompt_token_count=prompt_token_count,
            verify_prefill_ms=float(init_payload.get("verify_prefill_ms", 0.0)),
            **record_options,
        )
        session = build_verifier_session(
            request,
            prompt_ids=prompt_ids,
            prompt_token_count=prompt_token_count,
            shared_state=shared_state,
            core_session=init_payload.get("core_session"),
        )
        return self.after_start_session(
            session,
            init_payload,
            draft_session=draft_session,
        )

    def empty_verify_payload(self, shared_state: dict[str, Any], draft_output: dict[str, Any]):
        """Return a terminal verifier payload for empty or invalid drafts."""
        payload = {
            "accepted_length": 0,
            "accepted_ids": [],
            "corrected_ids": [],
            "rejected_draft_ids": [],
            "cloud_compute_ms": 0.0,
            "network_overhead_ms": 0.0,
            "stop": True,
            "stop_reason": shared_state.get("stop_reason") or "completion_limit",
            "progress": True,
            "round_stats": {
                "draft_count": 0,
                "accepted_length": 0,
                "corrected_count": 0,
                "rejected_draft_count": 0,
                "stop_reason": shared_state.get("stop_reason") or "completion_limit",
            },
        }
        payload["feedback"] = build_feedback_payload(draft_output, payload)
        return payload

    def normalize_draft_ids(self, draft_output: dict[str, Any], draft_ids=None) -> list[int]:
        """Resolve draft ids from Ianvs args or drafter output."""
        if draft_ids is None:
            return list(draft_output.get("draft_ids", []) or [])
        return list(draft_ids or [])

    def after_verify(
        self,
        session: dict[str, Any],
        draft_output: dict[str, Any],
        verify_payload: dict[str, Any],
        response: dict[str, Any],
    ):
        """Optional hook after common verify accounting."""
        del session, draft_output, verify_payload, response

    @staticmethod
    def _payload_from_verify_result(result: VerifyResult) -> dict[str, Any]:
        """Convert a typed verify result into the common payload dictionary."""
        if isinstance(result, VerifyResult):
            payload = dict(result.data or {})
            payload["accepted_ids"] = list(result.accepted_ids or [])
            payload["corrected_ids"] = list(result.corrected_ids or [])
            payload["rejected_draft_ids"] = list(result.rejected_draft_ids or [])
            payload["cloud_compute_ms"] = float(result.cloud_compute_ms or 0.0)
            payload["stop"] = bool(result.stop)
            payload["stop_reason"] = str(result.stop_reason or "")
            payload["token_provenance"] = list(result.token_provenance or [])
            payload["round_stats"] = dict(result.round_stats or {})
            return payload
        raise TypeError(
            "Verifier verify must return VerifyResult, "
            f"got {type(result)!r}."
        )

    def _run_verify_pipeline(
        self,
        *,
        session,
        draft_output=None,
        draft_ids=None,
        user_verify_fn,
        extra_kwargs: dict[str, Any] | None = None,
    ):
        """Run common verify/network/metric bookkeeping around user verification."""
        del extra_kwargs
        draft_output = dict(draft_output or {})
        shared_state = session["_shared_state"]
        resolved_draft_ids = self.normalize_draft_ids(draft_output, draft_ids=draft_ids)
        if self.should_return_empty_verify(draft_output, resolved_draft_ids):
            return self.empty_verify_payload(shared_state, draft_output)
        result = user_verify_fn(
            self,
            session,
            draft_output=draft_output,
            draft_ids=resolved_draft_ids,
        )
        verify_payload = self._payload_from_verify_result(result)
        accepted_ids = list(result.accepted_ids or [])
        corrected_ids = list(result.corrected_ids or [])
        rejected_ids = list(result.rejected_draft_ids or [])
        extra_fields = dict(result.extra_fields or {})
        if result.data:
            extra_fields = {
                **dict(extra_fields or {}),
                "algorithm_payload": dict(result.data),
            }
        response = apply_verify_payloads(
            self,
            shared_state,
            transport_payloads=list(result.payloads or []),
            draft_ids=resolved_draft_ids,
            accepted_ids=accepted_ids,
            corrected_ids=corrected_ids,
            rejected_ids=rejected_ids,
            verify_payload=verify_payload,
            draft_output=draft_output,
            token_provenance=list(result.token_provenance or []),
            round_stats=dict(result.round_stats or {}),
            extra_fields=extra_fields,
        )
        self.after_verify(session, draft_output, verify_payload, response)
        return response

    def verify(self, session, draft_output=None, draft_ids=None, **kwargs):
        """Verify one draft payload and update collaboration-shared statistics."""
        del session, draft_output, draft_ids, kwargs
        raise NotImplementedError(
            "Verifier subclasses must implement `verify` with @specdec_verify "
            "and return VerifyResult."
        )

    def should_return_empty_verify(
        self,
        draft_output: dict[str, Any],
        draft_ids: list[int],
    ) -> bool:
        """Return whether the verifier should emit a terminal empty payload."""
        del draft_output
        return not draft_ids

    def close_session(self, session, request=None):
        """Close one verifier request scope."""
        del session, request
        return None

    def encode_cloud_prompt(self, request: dict[str, Any]) -> dict[str, Any]:
        """Encode a cloud-only prompt payload."""
        prompt_ids = self.encode_prompt(request)
        return {"prompt_ids": prompt_ids}

    def cloud_generate(
        self,
        request: dict[str, Any],
        prompt_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Run cloud-only generation for algorithms that support it."""
        raise ValueError(f"{self.algorithm_name} does not support cloud-only inference.")

    def cloud_prompt_token_count(self, prompt_payload: dict[str, Any]) -> int:
        """Return cloud-only prompt token count."""
        return int(prompt_payload["prompt_ids"].shape[1])

    def cloud_extra_fields(self, *, generation, total_ms, network):
        """Return optional extra fields for cloud-only responses."""
        del generation, total_ms, network
        return None

    @torch.no_grad()
    def inference(self, data, token_callback=None, **kwargs):
        """Run cloud-only autoregressive decoding through the common runtime."""
        del token_callback, kwargs
        if not self.supports_cloud_only:
            raise ValueError(f"{self.algorithm_name} does not support cloud-only inference.")
        return run_cloud_only_generation(
            self,
            data,
            encode_prompt=self.encode_cloud_prompt,
            generate=self.cloud_generate,
            prompt_token_count=self.cloud_prompt_token_count,
            algorithm=self.cloud_only_algorithm,
            extra_fields_builder=self.cloud_extra_fields,
        )
