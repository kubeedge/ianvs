"""Block drafter module for the Ianvs benchmark example."""

from __future__ import annotations

import os
import sys

import torch
from sedna.common.class_factory import ClassFactory, ClassType
from transformers import DynamicCache

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
SPECDEC_DIR = os.path.dirname(os.path.dirname(MODULE_DIR))
for path in (MODULE_DIR, SPECDEC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from base_drafter import BaseSpeculativeDrafter
from common.modeling import crop_past_key_values
from common.decorators import specdec_draft
from common.schema import DraftResult
from common.timing import now
from copied_dflash_model import DFlashDraftModel
from copied_dflash_model import sample

os.environ["BACKEND_TYPE"] = "TORCH"


def _module_device(module, fallback):
    """Infer the device of one torch module."""
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device(fallback)


@ClassFactory.register(ClassType.GENERAL, alias="SpeculativeBlockDraftModel")
class SpeculativeBlockDraftModel(BaseSpeculativeDrafter):
    """DFlash-style block drafter.

    The common drafter base owns Ianvs lifecycle and final response handling.
    This class only implements block-specific state consumption and block
    drafting from verifier-provided target hidden states.
    """

    algorithm_name = "block_spec"
    default_model = "z-lab/Qwen3-8B-DFlash-b16"
    default_draft_tokens_per_step = 16
    allowed_modes = {"collaboration", "cloud-only"}
    collaboration_mode_name = "block-speculative-decoding"
    collaboration_algorithm = "block"
    cleanup_core_attrs = ("model", "embed_tokens", "lm_head", "tokenizer")

    def __init__(self, **kwargs):
        self.attn_implementation = str(kwargs.get("attn_implementation", "sdpa") or "sdpa")
        super().__init__(**kwargs)

    def build_core(self):
        """The block drafter implements model execution directly in this module."""
        return self

    def load_core(self):
        """Load the block draft model."""
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = DFlashDraftModel.from_pretrained(
            self.model_name,
            attn_implementation=self.attn_implementation,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.model_name.startswith("/"),
        ).to(self.device).eval()
        self.embed_tokens = None
        self.lm_head = None
        self.tokenizer = None
        self.embed_device = torch.device(self.device)
        self.lm_head_device = torch.device(self.device)

    def after_cleanup(self):
        """Reset inferred target-op devices after cleanup."""
        self.embed_device = torch.device("cpu")
        self.lm_head_device = torch.device("cpu")

    def attach_target_ops(self, embed_tokens, lm_head, tokenizer):
        """Attach target-side embedding and LM head used during block drafting."""
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head
        self.tokenizer = tokenizer
        self.embed_device = _module_device(embed_tokens, self.device)
        self.lm_head_device = _module_device(lm_head, self.device)

    def build_draft_state(self, verify_state):
        """Create one draft-side runtime state from verifier prefill state."""
        target_hidden = verify_state.get("target_hidden")
        return {
            "max_length": int(verify_state["max_length"]),
            "block_size": int(verify_state["block_size"]),
            "start": int(verify_state["start"]),
            "output_ids": verify_state["output_ids"].to(self.device),
            "position_ids": verify_state["position_ids"].to(self.device),
            "past_key_values_draft": DynamicCache(),
            "target_hidden": None if target_hidden is None else target_hidden.to(self.device),
        }

    def sync_draft_state(self, draft_state, verify_state):
        """Refresh the draft-side runtime state after one verifier round."""
        draft_state["max_length"] = int(verify_state["max_length"])
        draft_state["block_size"] = int(verify_state["block_size"])
        draft_state["start"] = int(verify_state["start"])
        draft_state["output_ids"] = verify_state["output_ids"].to(self.device)
        draft_state["position_ids"] = verify_state["position_ids"].to(self.device)
        target_hidden = verify_state.get("target_hidden")
        draft_state["target_hidden"] = None if target_hidden is None else target_hidden.to(self.device)
        return draft_state

    @torch.inference_mode()
    def draft_block(self, state, temperature=0.0):
        """Draft one non-causal block from latest target hidden states."""
        block_size = state["block_size"]
        start = state["start"]
        if block_size <= 1:
            return {
                "block_output_ids": state["output_ids"][:, start : start + block_size].clone(),
                "draft_count": 0,
                "draft_ids": [],
                "edge_compute_ms": 0.0,
            }

        block_output_ids = state["output_ids"][:, start : start + block_size].clone()
        position_ids = state["position_ids"][
            :, state["past_key_values_draft"].get_seq_length() : start + block_size
        ]
        draft_start = now(self.device)
        with torch.no_grad():
            ops_input_ids = block_output_ids.to(self.embed_device)
            noise_embedding = self.embed_tokens(ops_input_ids)
            if noise_embedding.device != torch.device(self.device):
                noise_embedding = noise_embedding.to(self.device)
            hidden_states = self.model(
                target_hidden=state["target_hidden"],
                noise_embedding=noise_embedding,
                position_ids=position_ids,
                past_key_values=state["past_key_values_draft"],
                use_cache=True,
                is_causal=False,
            )
            hidden_slice = hidden_states[:, -block_size + 1 :, :]
            draft_logits = self.lm_head(hidden_slice.to(self.lm_head_device))
            state["past_key_values_draft"] = crop_past_key_values(
                state["past_key_values_draft"],
                start,
            )
            drafted_ids = sample(draft_logits, temperature)
            block_output_ids[:, 1:] = drafted_ids.to(block_output_ids.device)
        return {
            "block_output_ids": block_output_ids,
            "draft_ids": drafted_ids[0].detach().cpu().tolist(),
            "draft_count": drafted_ids.shape[1],
            "edge_compute_ms": (now(self.device) - draft_start) * 1000.0,
        }

    def encode_prompt(self, request):
        """Block drafter has no independent tokenizer in collaboration mode."""
        del request
        raise ValueError("Block/DFlash drafter does not encode prompts independently.")

    def prepare_prompt(self, request):
        """Create a shared request shell; verifier performs target prompt encoding."""
        del request
        return {
            "prompt_ids": None,
            "prompt_token_count": 0,
        }

    def create_draft_session(self, prompt_payload):
        """Block drafter state is initialized after verifier target prefill."""
        del prompt_payload
        return None

    def shared_state_extra(self):
        """Add block runtime slots to shared collaboration state."""
        return {
            "decode_tokenizer": None,
            "prefill_network": {},
            "_block_draft_state": None,
            "_block_verify_state": None,
            "_block_draft_module": self,
        }

    @specdec_draft
    def step(self, session, *, window, feedback=None):
        """Run one block draft round.

        Block drafter window is fixed by the verifier-initialized block state,
        so the generic token-window resolver is intentionally bypassed here.
        """
        del window, feedback
        shared_state = session["_shared_state"]
        state = shared_state.get("_block_draft_state")
        if state is None:
            raise RuntimeError("Block verifier session has not initialized the block draft state.")
        if state["start"] >= state["max_length"]:
            return DraftResult(
                draft_ids=[],
                data={"draft_count": 0},
                edge_compute_ms=0.0,
            )
        hidden_bytes = 0
        if state.get("target_hidden") is not None:
            hidden_bytes = int(state["target_hidden"].numel()) * int(state["target_hidden"].element_size())
        payload = self.draft_block(state, temperature=self.temperature)
        return DraftResult(
            draft_ids=list(payload.get("draft_ids", []) or []),
            data={
                "block_output_ids": payload.get("block_output_ids"),
                "draft_count": int(payload.get("draft_count", 0) or 0),
                "downlink_hidden_bytes": hidden_bytes,
            },
            edge_compute_ms=float(payload.get("edge_compute_ms", 0.0) or 0.0),
        )

    def collaboration_extra_fields(self, shared_state):
        """Add block timing details to the final collaboration response."""
        return {
            "prefill_network": dict(shared_state.get("prefill_network", {})),
            "time_breakdown": {
                "prefill_compute_ms": round(float(shared_state.get("prefill_ms", 0.0)), 6),
                "edge_decode_compute_ms": round(float(shared_state.get("edge_compute_ms", 0.0)), 6),
                "cloud_verify_compute_ms": round(float(shared_state.get("cloud_compute_ms", 0.0)), 6),
                "network_ms": round(float(shared_state.get("network_ms", 0.0)), 6),
                "network_propagation_ms": round(float(shared_state.get("network_propagation_ms", 0.0)), 6),
                "network_transfer_ms": round(float(shared_state.get("network_transfer_ms", 0.0)), 6),
            },
        }
