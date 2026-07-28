"""AR drafter module for the Ianvs benchmark example."""

import os
import sys
from dataclasses import dataclass, field

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
SPECDEC_DIR = os.path.dirname(os.path.dirname(MODULE_DIR))
for path in (MODULE_DIR, SPECDEC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch
from sedna.common.class_factory import ClassFactory, ClassType
from transformers import AutoModelForCausalLM, AutoTokenizer

from base_drafter import BaseSpeculativeDrafter
from common.runtime import build_prompt_text
from common.modeling import crop_past_key_values
from common.config import as_int
from common.decorators import specdec_draft
from common.modeling import build_topk_distribution
from common.modeling import sample_token_id
from common.modeling import sample_token_id_from_topk
from common.schema import DraftResult
from common.timing import now

os.environ["BACKEND_TYPE"] = "TORCH"


@dataclass
class ARDraftState:
    """Mutable drafter-side KV state for one request."""

    prompt_ids: torch.Tensor
    prompt_token_count: int
    past_key_values: object
    last_logits: torch.Tensor | None
    prefill_ms: float
    cached_ids: list[int] = field(default_factory=list)
    pending_ids: list[int] = field(default_factory=list)


@ClassFactory.register(ClassType.GENERAL, alias="SpeculativeDraftModel")
class SpeculativeDraftModel(BaseSpeculativeDrafter):
    """Token-level AR drafter.

    Public Ianvs lifecycle, collaboration finalization, edge-only result
    formatting, cleanup, and shared state management are inherited from
    `BaseSpeculativeDrafter`. This class only implements AR-specific model
    setup and draft behavior.
    """

    algorithm_name = "ar_spec"
    default_model = "Qwen/Qwen2.5-0.5B-Instruct"
    default_draft_tokens_per_step = 8
    allowed_modes = {"collaboration", "cloud-only", "edge-only"}
    export_model_path = True
    supports_edge_only = True
    collaboration_mode_name = "token-level-speculative-decoding"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.draft_top_k = max(0, as_int(kwargs.get("draft_top_k"), 0))

    def build_core(self):
        """The AR drafter implements model execution directly in this module."""
        return self

    def load_core(self):
        """Load tokenizer and small causal LM."""
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.model_name.startswith("/"),
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.model_name.startswith("/"),
        ).to(self.device)
        self.model.eval()

    def encode_prompt(self, request):
        """Tokenize one normalized request prompt."""
        prompt_text = build_prompt_text(self, self.tokenizer, request)
        encoded = self.tokenizer(prompt_text, return_tensors="pt")
        return encoded.input_ids.to(self.device)

    def create_draft_session(self, prompt_payload):
        """Prefill the small model for collaboration."""
        prompt_ids = prompt_payload["prompt_ids"]
        start = now(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=prompt_ids, use_cache=True)
        return ARDraftState(
            prompt_ids=prompt_ids,
            prompt_token_count=int(prompt_ids.shape[1]),
            past_key_values=outputs.past_key_values,
            last_logits=outputs.logits[:, -1, :],
            prefill_ms=(now(self.device) - start) * 1000.0,
        )

    def _flush_pending(self, state):
        """Append deferred committed tokens into the cached prefix."""
        if not state.pending_ids:
            return 0.0
        pending_tensor = torch.tensor([state.pending_ids], dtype=torch.long, device=self.device)
        start = now(self.device)
        with torch.no_grad():
            outputs = self.model(
                input_ids=pending_tensor,
                past_key_values=state.past_key_values,
                use_cache=True,
            )
        state.past_key_values = outputs.past_key_values
        state.last_logits = outputs.logits[:, -1, :]
        state.cached_ids.extend(state.pending_ids)
        state.pending_ids = []
        return (now(self.device) - start) * 1000.0

    def consume_feedback(self, session, feedback=None):
        """Apply verifier feedback to the draft-side KV session."""
        feedback = feedback or session.pop("_pending_feedback", None)
        if not feedback:
            return None
        draft_output = dict((feedback or {}).get("draft_output", {}) or {})
        verify_output = dict((feedback or {}).get("verify_output", {}) or {})
        state = session["_core_session"]
        base_cached_count = int(draft_output.get("base_cached_count", len(state.cached_ids)))
        accepted_ids = list(verify_output.get("accepted_ids", []) or [])
        corrected_ids = list(verify_output.get("corrected_ids", []) or [])
        keep_count = base_cached_count + len(accepted_ids)
        cache_length = state.prompt_token_count + keep_count
        state.past_key_values = crop_past_key_values(state.past_key_values, cache_length)
        state.cached_ids = state.cached_ids[:keep_count]
        state.pending_ids = list(corrected_ids)
        state.last_logits = None if corrected_ids else state.last_logits
        session["_last_feedback"] = verify_output
        return verify_output

    def empty_draft_payload(self, session):
        """Return an AR-shaped empty draft payload."""
        return {
            "draft_ids": [],
            "draft_logits": [],
            "edge_compute_ms": 0.0,
            "base_cached_count": len(session["_core_session"].cached_ids),
        }

    @specdec_draft
    def step(self, session, *, window, feedback=None):
        """Run one AR draft round."""
        del feedback
        state = session["_core_session"]
        flush_ms = self._flush_pending(state)
        if state.last_logits is None:
            raise RuntimeError("Drafter session has no next-token logits after pending flush.")

        draft_ids = []
        draft_logits = []
        base_cached_count = len(state.cached_ids)
        start = now(self.device)
        cache = state.past_key_values
        next_logits = state.last_logits
        use_topk = float(self.temperature or 0.0) >= 1e-5 and int(self.draft_top_k or 0) > 0
        with torch.no_grad():
            for _ in range(window):
                if use_topk:
                    topk_distribution = build_topk_distribution(
                        next_logits,
                        self.temperature,
                        self.draft_top_k,
                    )
                    draft_logits.append(topk_distribution)
                    token_id = sample_token_id_from_topk(topk_distribution)
                else:
                    draft_logits.append(next_logits.detach().cpu())
                    token_id = sample_token_id(next_logits, self.temperature)
                draft_ids.append(token_id)
                step_ids = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
                outputs = self.model(input_ids=step_ids, past_key_values=cache, use_cache=True)
                cache = outputs.past_key_values
                next_logits = outputs.logits[:, -1, :]

        state.past_key_values = cache
        state.last_logits = next_logits
        state.cached_ids.extend(draft_ids)
        return DraftResult(
            draft_ids=draft_ids,
            data={
                "draft_logits": draft_logits,
                "base_cached_count": base_cached_count,
            },
            edge_compute_ms=(now(self.device) - start) * 1000.0 + flush_ms,
        )

    def before_close_session(self, session):
        """Consume any final feedback before collaboration finalization."""
        self.consume_feedback(
            session,
            feedback=session.pop("_pending_feedback", None),
        )

    def edge_generate(self, prompt_payload, completion_limit):
        """Generate one edge-only AR completion."""
        generated = []
        token_timestamps_ms = []
        start = now(self.device)
        ttft_ms = None
        with torch.no_grad():
            outputs = self.model(input_ids=prompt_payload["prompt_ids"], use_cache=True)
            prefill_ms = (now(self.device) - start) * 1000.0
            cache = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]
            for _ in range(completion_limit):
                token_id = sample_token_id(next_logits, self.temperature)
                generated.append(token_id)
                token_time_ms = (now(self.device) - start) * 1000.0
                token_timestamps_ms.append(token_time_ms)
                if ttft_ms is None:
                    ttft_ms = token_time_ms
                if self.stop_on_eos and token_id == self.tokenizer.eos_token_id:
                    break
                step_ids = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
                outputs = self.model(input_ids=step_ids, past_key_values=cache, use_cache=True)
                cache = outputs.past_key_values
                next_logits = outputs.logits[:, -1, :]
        total_ms = (now(self.device) - start) * 1000.0
        return {
            "completion_ids": generated,
            "prefill_ms": prefill_ms if completion_limit > 0 else total_ms,
            "ttft_ms": ttft_ms or total_ms,
            "total_ms": total_ms,
            "token_timestamps_ms": token_timestamps_ms,
            "stop_reason": "eos" if generated and generated[-1] == self.tokenizer.eos_token_id else "completion_limit",
        }
