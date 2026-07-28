"""Template verifier module for adding a new speculative decoding algorithm.

Copy this file into `algorithms/<your_algorithm>/verifier.py`, register a new
ClassFactory alias, and replace the hook bodies with real verification logic.
"""

from __future__ import annotations

from sedna.common.class_factory import ClassFactory, ClassType

from base_verifier import BaseSpeculativeVerifier
from common.decorators import specdec_verify
from common.payload import control_payload
from common.payload import token_payload
from common.schema import VerifyResult


@ClassFactory.register(ClassType.GENERAL, alias="TemplateVerifyModel")
class TemplateVerifyModel(BaseSpeculativeVerifier):
    """Minimal verifier implementation shape."""

    algorithm_name = "template_spec"
    default_model = "replace-with-target-model"
    default_draft_tokens_per_step = 8

    def build_core(self):
        """Build or return the verifier-side model object."""
        return None

    def load_core(self):
        """Load verifier resources and set `self.model` / `self.tokenizer`."""
        self.model = None
        self.tokenizer = None

    def encode_prompt(self, request):
        """Return device-local prompt ids for `request['query']`."""
        raise NotImplementedError

    def init_verify_state(self, request, prompt_ids, draft_session=None, shared_state=None):
        """Create verifier request state and report prefill timing."""
        del request, prompt_ids, draft_session, shared_state
        return {
            "core_session": {},
            "verify_prefill_ms": 0.0,
        }

    @specdec_verify
    def verify(self, session, *, draft_output, draft_ids):
        """Return one verify result.

        `payloads` declares logical transfers for automatic network accounting.
        `data` is preserved unchanged in the verifier payload.
        """
        del session, draft_output
        accepted_ids = list(draft_ids[:1])
        rejected_ids = list(draft_ids[1:])
        return VerifyResult(
            accepted_ids=accepted_ids,
            corrected_ids=[],
            rejected_draft_ids=rejected_ids,
            payloads=[
                token_payload(draft_ids, "edge_to_cloud", "draft_ids"),
                token_payload(accepted_ids, "cloud_to_edge", "accepted_ids"),
                token_payload([], "cloud_to_edge", "corrected_ids"),
                token_payload(rejected_ids, "cloud_to_edge", "rejected_draft_ids"),
                control_payload("cloud_to_edge", "round_control"),
            ],
            data={"custom_verify_payload": {"example": True}},
            cloud_compute_ms=0.0,
            stop=True,
            stop_reason="template_stop",
            round_stats={
                "draft_count": len(draft_ids),
                "accepted_length": len(accepted_ids),
                "corrected_count": 0,
                "rejected_draft_count": len(rejected_ids),
                "stop_reason": "template_stop",
            },
        )
