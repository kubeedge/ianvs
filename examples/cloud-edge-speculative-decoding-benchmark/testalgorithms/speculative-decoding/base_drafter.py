"""Reusable Ianvs drafter module base for speculative decoding algorithms."""

from __future__ import annotations

from typing import Any

from common.runtime import build_collaboration_result
from common.runtime import cleanup_model_runtime
from common.runtime import close_shared_collaboration_session
from common.runtime import decode_tokens
from common.runtime import init_model_runtime
from common.runtime import run_edge_only_generation
from common.runtime import start_drafter_session
from common.schema import DraftResult


class BaseSpeculativeDrafter:
    """Common Ianvs-facing drafter lifecycle.

    Subclasses should only provide algorithm-specific hooks: model creation,
    prompt encoding, draft-state initialization, one draft step, and optional
    edge-only generation. The base class owns Ianvs session wiring, shared
    state creation, edge-only result formatting, collaboration finalization,
    and cleanup.
    """

    algorithm_name = "base"
    role = "drafter"
    default_model = ""
    default_draft_tokens_per_step = 8
    allowed_modes: set[str] | None = None
    default_mode = "collaboration"
    export_model_path = False
    supports_edge_only = False
    collaboration_mode_name = "speculative-decoding"
    collaboration_algorithm: str | None = None
    cleanup_core_attrs = ("model", "tokenizer")

    def __init__(self, **kwargs):
        init_model_runtime(
            self,
            kwargs,
            default_model=self.default_model,
            default_draft_tokens_per_step=self.default_draft_tokens_per_step,
            allowed_modes=self.allowed_modes,
            default_mode=self.default_mode,
            export_model_path=self.export_model_path,
        )
        self.core = self.build_core()

    def build_core(self):
        """Build the algorithm-specific drafter implementation."""
        raise NotImplementedError

    def load_core(self):
        """Load algorithm-specific drafter resources."""
        self.core.load()
        self.tokenizer = getattr(self.core, "tokenizer", None)
        self.model = getattr(self.core, "model", None)

    def load(self, *args, **kwargs):
        """Load the draft model."""
        del args, kwargs
        self.load_core()

    def cleanup(self):
        """Release loaded resources and request sessions."""
        cleanup_model_runtime(self, *self.cleanup_core_attrs)
        self.after_cleanup()

    def after_cleanup(self):
        """Optional subclass hook after common resource cleanup."""

    def decode_tokens(self, token_ids, skip_special_tokens=True):
        """Decode token ids using the attached tokenizer when available."""
        return decode_tokens(self, token_ids, skip_special_tokens=skip_special_tokens)

    def prepare_prompt(self, request: dict[str, Any]) -> dict[str, Any]:
        """Encode one normalized request prompt."""
        prompt_ids = self.encode_prompt(request)
        return {
            "prompt_ids": prompt_ids,
            "prompt_token_count": int(prompt_ids.shape[1]),
        }

    def encode_prompt(self, request: dict[str, Any]):
        """Return device-local prompt ids for a normalized request."""
        raise NotImplementedError

    def create_draft_session(self, prompt_payload: dict[str, Any]):
        """Create algorithm-specific per-request draft state."""
        return None

    def shared_state_extra(self) -> dict[str, Any] | None:
        """Return extra fields inserted into shared collaboration state."""
        return None

    def after_start_session(self, session: dict[str, Any]) -> dict[str, Any]:
        """Optional subclass hook after common session creation."""
        return session

    def start_session(self, data=None, request=None, **kwargs):
        """Create one draft-side collaboration session."""
        del kwargs
        session = start_drafter_session(
            self,
            data=data,
            request=request,
            prepare_prompt=self.prepare_prompt,
            create_core_session=self.create_draft_session,
            shared_state_extra=self.shared_state_extra(),
        )
        return self.after_start_session(session)

    def consume_feedback(self, session: dict[str, Any], feedback=None):
        """Apply verifier feedback before the next draft step."""
        del session, feedback
        return None

    def resolve_draft_window(
        self,
        session: dict[str, Any],
        max_draft_tokens: int | None = None,
    ) -> int:
        """Resolve how many draft tokens the current round may propose."""
        shared_state = session["_shared_state"]
        remaining = max(
            int(shared_state["completion_limit"]) - len(shared_state["committed_ids"]),
            0,
        )
        window = min(self.draft_tokens_per_step, remaining)
        if max_draft_tokens is not None:
            window = min(max(int(max_draft_tokens or 0), 0), remaining)
        return int(window)

    def empty_draft_payload(self, session: dict[str, Any]) -> dict[str, Any]:
        """Return a no-progress draft payload for terminal requests."""
        del session
        return {
            "draft_ids": [],
            "edge_compute_ms": 0.0,
        }

    @staticmethod
    def _payload_from_draft_result(result: DraftResult) -> dict[str, Any]:
        """Convert a user draft result into the Ianvs payload dictionary."""
        if isinstance(result, DraftResult):
            payload = dict(result.data or {})
            payload["draft_ids"] = list(result.draft_ids or [])
            payload["edge_compute_ms"] = float(result.edge_compute_ms or 0.0)
            if result.stop:
                payload["stop"] = True
                payload["stop_reason"] = str(result.stop_reason or "")
            return payload
        raise TypeError(
            "Drafter step must return DraftResult, "
            f"got {type(result)!r}."
        )

    def _run_draft_pipeline(
        self,
        *,
        session,
        feedback=None,
        max_draft_tokens=None,
        user_step_fn,
        extra_kwargs: dict[str, Any] | None = None,
    ):
        """Run common draft bookkeeping around a user draft function."""
        del extra_kwargs
        self.consume_feedback(session, feedback=feedback)
        window = self.resolve_draft_window(session, max_draft_tokens=max_draft_tokens)
        if window <= 0:
            return self.empty_draft_payload(session)
        result = user_step_fn(
            self,
            session,
            feedback=feedback,
            window=window,
        )
        payload = self._payload_from_draft_result(result)
        session["_shared_state"]["edge_compute_ms"] += float(
            payload.get("edge_compute_ms", 0.0) or 0.0
        )
        payload["selected_window"] = int(window)
        return payload

    def step(self, session, feedback=None, max_draft_tokens=None, **kwargs):
        """Run one speculative draft round."""
        del session, feedback, max_draft_tokens, kwargs
        raise NotImplementedError(
            "Drafter subclasses must implement `step` with @specdec_draft "
            "and return DraftResult."
        )

    def collaboration_extra_fields(self, shared_state: dict[str, Any]) -> dict[str, Any] | None:
        """Return extra final collaboration response fields."""
        del shared_state
        return None

    def before_close_session(self, session: dict[str, Any]):
        """Optional hook before collaboration finalization."""

    def close_session(self, session, request=None):
        """Finalize one collaboration request."""
        del request
        return close_shared_collaboration_session(
            self,
            session,
            build_result=lambda shared_state: build_collaboration_result(
                self,
                shared_state,
                mode=self.collaboration_mode_name,
                algorithm=self.collaboration_algorithm,
                extra_fields=self.collaboration_extra_fields(shared_state),
            ),
            pre_close=self.before_close_session,
        )

    def edge_generate(self, prompt_payload: dict[str, Any], completion_limit: int):
        """Run edge-only generation for algorithms that support it."""
        raise ValueError(f"{self.algorithm_name} does not support edge-only inference.")

    def inference(self, data=None, request=None, **kwargs):
        """Run edge-only inference through the common runtime."""
        del kwargs
        if not self.supports_edge_only:
            raise ValueError(f"{self.algorithm_name} does not support edge-only inference.")
        return run_edge_only_generation(
            self,
            data,
            request=request,
            encode_prompt=self.prepare_prompt,
            generate=self.edge_generate,
            algorithm=self.collaboration_algorithm,
        )
