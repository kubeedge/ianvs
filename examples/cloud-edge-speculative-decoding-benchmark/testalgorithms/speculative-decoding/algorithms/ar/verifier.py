"""AR verifier module for the Ianvs benchmark example."""

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

from base_verifier import BaseSpeculativeVerifier
from common.runtime import build_prompt_text
from common.modeling import crop_past_key_values
from common.config import as_int
from common.decorators import specdec_verify
from common.payload import draft_distribution_payloads
from common.payload import token_payload
from common.modeling import greedy_token_id
from common.modeling import is_topk_probability_payload
from common.modeling import lookup_topk_probability
from common.modeling import probs_from_logits
from common.modeling import sample_from_probs
from common.modeling import sample_from_residual
from common.modeling import sample_from_sparse_residual
from common.modeling import sample_token_id
from common.schema import VerifyResult
from common.timing import now

os.environ["BACKEND_TYPE"] = "TORCH"


@dataclass
class ARVerifyState:
    """Mutable verifier-side KV state for one request."""

    prompt_ids: torch.Tensor
    prompt_token_count: int
    past_key_values: object
    last_logits: torch.Tensor | None
    prefill_ms: float
    cached_ids: list[int] = field(default_factory=list)
    pending_ids: list[int] = field(default_factory=list)


@ClassFactory.register(ClassType.GENERAL, alias="SpeculativeVerifyModel")
class SpeculativeVerifyModel(BaseSpeculativeVerifier):
    """Token-level AR verifier.

    Public Ianvs lifecycle, network simulation, metrics, response formatting,
    cloud-only runtime, and cleanup are inherited from `BaseSpeculativeVerifier`.
    This class only implements AR-specific verify behavior and payload
    declaration.
    """

    algorithm_name = "ar_spec"
    default_model = "Qwen/Qwen2.5-7B-Instruct"
    default_draft_tokens_per_step = 8

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.draft_top_k = max(0, as_int(kwargs.get("draft_top_k"), 0))

    def build_core(self):
        """The AR verifier implements model execution directly in this module."""
        return self

    def load_core(self):
        """Load tokenizer and large causal LM."""
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
        """Encode one AR prompt."""
        prompt_text = build_prompt_text(self, self.tokenizer, request)
        encoded = self.tokenizer(prompt_text, return_tensors="pt")
        prompt_ids = encoded.input_ids.to(self.device)
        return prompt_ids

    def resolve_prompt_ids(self, request, draft_session=None):
        """Reuse drafter prompt ids when collaboration provides them."""
        if draft_session is not None:
            return draft_session["prompt_ids"].to(self.device)
        return self.encode_prompt(request)

    @specdec_verify
    def verify(self, session, *, draft_output, draft_ids):
        """Run one AR verify round."""
        state = session["_core_session"]
        draft_logits = draft_output.get("draft_logits", [])
        max_new_tokens = session["_shared_state"]["completion_limit"]
        if state.last_logits is None and not state.pending_ids:
            raise RuntimeError("Verifier session has no next-token logits before verification.")

        pending_count = len(state.pending_ids)
        combined_ids = list(state.pending_ids) + list(draft_ids)
        if not combined_ids:
            return VerifyResult(
                accepted_ids=[],
                corrected_ids=[],
                rejected_draft_ids=[],
                cloud_compute_ms=0.0,
                stop=True,
                stop_reason="stalled",
                token_provenance=[],
                round_stats={
                    "draft_count": 0,
                    "accepted_length": 0,
                    "corrected_count": 0,
                    "rejected_draft_count": 0,
                    "stop_reason": "stalled",
                    "rejected_draft_tokens": [],
                },
            )

        combined_tensor = torch.tensor([combined_ids], dtype=torch.long, device=self.device)
        base_cached_count = len(state.cached_ids)
        start = now(self.device)
        with torch.no_grad():
            outputs = self.model(
                input_ids=combined_tensor,
                past_key_values=state.past_key_values,
                use_cache=True,
            )
        combined_logits = outputs.logits[0]
        next_logits_after_block = combined_logits[len(combined_ids) - 1].unsqueeze(0)

        accepted_ids = []
        corrected_ids = []
        rejected_draft_ids = []
        token_provenance = []
        stop = False
        stop_reason = ""
        rejection_logits = None

        for idx, token_id in enumerate(draft_ids):
            combined_pos = pending_count + idx
            if combined_pos == 0:
                p_logits = state.last_logits
            else:
                p_logits = combined_logits[combined_pos - 1].unsqueeze(0)
            if float(self.temperature or 0.0) < 1e-5:
                verifier_token = greedy_token_id(p_logits)
                if verifier_token == token_id:
                    accepted_ids.append(token_id)
                    continue
                corrected_ids = [verifier_token]
                rejected_draft_ids = draft_ids[idx:]
                rejection_logits = p_logits
                break
            p_probs = probs_from_logits(p_logits, self.temperature)
            if is_topk_probability_payload(draft_logits[idx]):
                q_token = max(
                    lookup_topk_probability(token_id, draft_logits[idx], p_probs.device),
                    1e-12,
                )
                accept_prob = min(1.0, float(p_probs[token_id].item()) / q_token)
            else:
                q_logits = draft_logits[idx].reshape(-1).float().to(p_probs.device)
                q_probs = torch.softmax(q_logits / max(float(self.temperature or 0.0), 1e-5), dim=-1)
                accept_prob = min(
                    1.0,
                    float(p_probs[token_id].item()) / max(float(q_probs[token_id].item()), 1e-12),
                )
            if torch.rand(1, device=p_probs.device).item() <= accept_prob:
                accepted_ids.append(token_id)
                continue
            if is_topk_probability_payload(draft_logits[idx]):
                corrected_ids = [sample_from_sparse_residual(p_probs, draft_logits[idx])]
            else:
                corrected_ids = [sample_from_residual(p_probs, q_probs)]
            rejected_draft_ids = draft_ids[idx:]
            rejection_logits = p_logits
            break

        logical_total_before = base_cached_count + pending_count
        if accepted_ids and self.stop_on_eos and accepted_ids[-1] == self.tokenizer.eos_token_id:
            stop = True
            stop_reason = "eos"
        elif not corrected_ids and len(accepted_ids) == len(draft_ids):
            logical_total = logical_total_before + len(accepted_ids)
            if logical_total < max_new_tokens:
                bonus = (
                    greedy_token_id(next_logits_after_block)
                    if float(self.temperature or 0.0) < 1e-5
                    else sample_from_probs(probs_from_logits(next_logits_after_block, self.temperature))
                )
                corrected_ids = [bonus]
            else:
                stop = True
                stop_reason = "completion_limit"

        if corrected_ids and self.stop_on_eos and corrected_ids[-1] == self.tokenizer.eos_token_id:
            stop = True
            stop_reason = "eos"
        elif accepted_ids and self.stop_on_eos and accepted_ids[-1] == self.tokenizer.eos_token_id:
            stop = True
            stop_reason = "eos"

        if not stop and logical_total_before + len(accepted_ids) + len(corrected_ids) >= max_new_tokens:
            stop = True
            stop_reason = "completion_limit"

        kept_cached_count = base_cached_count + pending_count + len(accepted_ids)
        cache_length = state.prompt_token_count + kept_cached_count
        state.past_key_values = crop_past_key_values(outputs.past_key_values, cache_length)
        state.cached_ids.extend(state.pending_ids)
        state.cached_ids.extend(accepted_ids)
        state.pending_ids = list(corrected_ids)
        state.last_logits = next_logits_after_block if rejection_logits is None else rejection_logits

        for token_id in accepted_ids:
            token_provenance.append(
                {
                    "token_id": token_id,
                    "token_text": self.tokenizer.decode([token_id], skip_special_tokens=False),
                    "source": "draft_accepted",
                }
            )
        for token_id in corrected_ids:
            source = "verifier_bonus" if len(accepted_ids) == len(draft_ids) else "verifier_correction"
            token_provenance.append(
                {
                    "token_id": token_id,
                    "token_text": self.tokenizer.decode([token_id], skip_special_tokens=False),
                    "source": source,
                }
            )

        return VerifyResult(
            accepted_ids=accepted_ids,
            corrected_ids=corrected_ids,
            rejected_draft_ids=rejected_draft_ids,
            payloads=[
                token_payload(draft_ids, "edge_to_cloud", "draft_ids"),
                *draft_distribution_payloads(draft_output.get("draft_logits", [])),
                token_payload(accepted_ids, "cloud_to_edge", "accepted_ids"),
                token_payload(corrected_ids, "cloud_to_edge", "corrected_ids"),
                token_payload(rejected_draft_ids, "cloud_to_edge", "rejected_draft_ids"),
            ],
            cloud_compute_ms=(now(self.device) - start) * 1000.0,
            stop=stop,
            stop_reason=stop_reason,
            token_provenance=token_provenance,
            round_stats={
                "draft_count": len(draft_ids),
                "accepted_length": len(accepted_ids),
                "corrected_count": len(corrected_ids),
                "rejected_draft_count": len(rejected_draft_ids),
                "stop_reason": stop_reason,
                "rejected_draft_tokens": [
                    {"token_id": tid, "token_text": self.tokenizer.decode([tid], skip_special_tokens=False)}
                    for tid in rejected_draft_ids
                ],
            },
        )

    def init_verify_state(self, request, prompt_ids, draft_session=None, shared_state=None):
        """Prefill prompt tokens and create persistent target KV state."""
        del request, draft_session, shared_state
        start = now(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids=prompt_ids, use_cache=True)
        state = ARVerifyState(
            prompt_ids=prompt_ids,
            prompt_token_count=int(prompt_ids.shape[1]),
            past_key_values=outputs.past_key_values,
            last_logits=outputs.logits[:, -1, :],
            prefill_ms=(now(self.device) - start) * 1000.0,
        )
        return {
            "core_session": state,
            "verify_prefill_ms": state.prefill_ms,
        }

    def encode_cloud_prompt(self, request):
        """Encode a cloud-only AR prompt."""
        prompt_text = build_prompt_text(self, self.tokenizer, request)
        encoded = self.tokenizer(prompt_text, return_tensors="pt")
        prompt_ids = encoded.input_ids.to(self.device)
        return {
            "prompt_ids": prompt_ids,
            "attention_mask": encoded.attention_mask.to(self.device),
            "transport_payload": token_payload(prompt_ids, "edge_to_cloud", "prompt_ids"),
            "completion_payload": lambda completion_ids: token_payload(
                completion_ids,
                "cloud_to_edge",
                "completion_ids",
            ),
        }

    def cloud_generate(self, request, prompt_payload):
        """Generate one cloud-only AR completion."""
        generated = []
        token_timestamps_ms = []
        start = now(self.device)
        ttft_ms = None
        with torch.no_grad():
            outputs = self.model(
                input_ids=prompt_payload["prompt_ids"],
                attention_mask=prompt_payload["attention_mask"],
                use_cache=True,
            )
            prefill_ms = (now(self.device) - start) * 1000.0
            cache = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]
            for _ in range(request["completion_tokens"]):
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
            "prefill_ms": prefill_ms if request["completion_tokens"] > 0 else total_ms,
            "ttft_ms": ttft_ms or total_ms,
            "total_ms": total_ms,
            "token_timestamps_ms": token_timestamps_ms,
            "stop_reason": "eos" if generated and generated[-1] == self.tokenizer.eos_token_id else "completion_limit",
        }
