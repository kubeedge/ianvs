"""Template drafter module for adding a new speculative decoding algorithm.

Copy this file into `algorithms/<your_algorithm>/drafter.py`, register a new
ClassFactory alias, and replace the hook bodies with real model logic.
"""

from __future__ import annotations

from sedna.common.class_factory import ClassFactory, ClassType

from base_drafter import BaseSpeculativeDrafter
from common.decorators import specdec_draft
from common.schema import DraftResult


@ClassFactory.register(ClassType.GENERAL, alias="TemplateDraftModel")
class TemplateDraftModel(BaseSpeculativeDrafter):
    """Minimal drafter implementation shape."""

    algorithm_name = "template_spec"
    default_model = "replace-with-draft-model"
    default_draft_tokens_per_step = 8
    allowed_modes = {"collaboration", "edge-only"}
    supports_edge_only = False
    collaboration_mode_name = "template-speculative-decoding"

    def build_core(self):
        """Build or return the drafter-side model object."""
        return None

    def load_core(self):
        """Load drafter resources and set `self.model` / `self.tokenizer` if needed."""
        self.model = None
        self.tokenizer = None

    def encode_prompt(self, request):
        """Return device-local prompt ids for `request['query']`."""
        raise NotImplementedError

    def create_draft_session(self, prompt_payload):
        """Create algorithm-specific request state after prompt encoding."""
        del prompt_payload
        return {}

    @specdec_draft
    def step(self, session, *, window, feedback=None):
        """Return one draft result.

        Extra `data` fields are passed unchanged to the verifier as
        `draft_output`.
        """
        del session, feedback
        return DraftResult(
            draft_ids=[0 for _ in range(window)],
            data={"custom_draft_payload": {"example": True}},
            edge_compute_ms=0.0,
        )
