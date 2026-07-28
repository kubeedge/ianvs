"""Block verifier module for the Ianvs benchmark example."""

from __future__ import annotations

import os
import sys

import torch
from sedna.common.class_factory import ClassFactory, ClassType
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.generation.streamers import BaseStreamer

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
SPECDEC_DIR = os.path.dirname(os.path.dirname(MODULE_DIR))
for path in (MODULE_DIR, SPECDEC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from base_verifier import BaseSpeculativeVerifier
from common.runtime import build_prompt_text
from common.modeling import crop_past_key_values
from common.decorators import specdec_verify
from common.payload import control_payload
from common.payload import custom_payload
from common.payload import tensor_payload
from common.payload import token_payload
from common.schema import VerifyResult
from common.timing import now
from copied_dflash_model import extract_context_feature
from copied_dflash_model import sample

os.environ["BACKEND_TYPE"] = "TORCH"


class _TokenTimingStreamer(BaseStreamer):
    """Capture generated token ids and arrival timestamps for `generate()`."""

    def __init__(self, prompt_token_count, start_time, device):
        self.prompt_token_count = int(prompt_token_count)
        self.start_time = float(start_time)
        self.device = device
        self.token_timestamps_ms = []
        self._prompt_skipped = False

    def put(self, value):
        """Record streamed generated tokens."""
        if isinstance(value, torch.Tensor):
            tokens = value.detach().cpu().reshape(-1).tolist()
        elif isinstance(value, (list, tuple)):
            tokens = [int(item) for item in value]
        else:
            tokens = [int(value)]

        if not self._prompt_skipped:
            if len(tokens) == self.prompt_token_count:
                self._prompt_skipped = True
                return
            self._prompt_skipped = True

        timestamp_ms = (now(self.device) - self.start_time) * 1000.0
        for _ in tokens:
            self.token_timestamps_ms.append(timestamp_ms)

    def end(self):
        """No-op end hook required by streamer interface."""
        return None


def _normalize_generation_backend(value):
    """Normalize the baseline backend name and reject removed backends."""
    backend = str(value or "custom").strip().lower().replace("-", "_")
    if backend not in {"custom", "transformers"}:
        raise ValueError(
            f"Unsupported generation backend: {value}. "
            f"Expected one of custom/transformers."
        )
    return backend


@ClassFactory.register(ClassType.GENERAL, alias="SpeculativeBlockVerifyModel")
class SpeculativeBlockVerifyModel(BaseSpeculativeVerifier):
    """DFlash-style block verifier.

    The common verifier base owns Ianvs lifecycle, network simulation, metrics,
    response formatting, and cloud-only execution. This class only implements
    block-specific target prefill, verification, and payload declaration.
    """

    algorithm_name = "block_spec"
    default_model = "Qwen/Qwen3-8B"
    default_draft_tokens_per_step = 16
    include_network_breakdown = True
    cloud_only_algorithm = "block"
    cleanup_core_attrs = ("model", "tokenizer", "target_layer_ids", "mask_token_id")

    def __init__(self, **kwargs):
        self.attn_implementation = str(kwargs.get("attn_implementation", "sdpa") or "sdpa")
        self.block_cloud_only_backend = _normalize_generation_backend(
            kwargs.get("block_cloud_only_backend", kwargs.get("generation_backend", "custom"))
        )
        super().__init__(**kwargs)

    def build_core(self):
        """The block verifier implements model execution directly in this module."""
        return self

    def load_core(self):
        """Load tokenizer and target model."""
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.model_name.startswith("/"),
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            attn_implementation=self.attn_implementation,
            torch_dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.model_name.startswith("/"),
        ).to(self.device).eval()
        self.target_layer_ids = None
        self.mask_token_id = None

    def encode_prompt(self, request):
        """Encode one block-path prompt."""
        input_text = build_prompt_text(self, self.tokenizer, request)
        return self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)

    def _attach_target_resources(self, draft_module):
        """Attach target-side operations required by the DFlash draft model."""
        draft_module.attach_target_ops(
            self.model.model.embed_tokens,
            self.model.lm_head,
            self.tokenizer,
        )
        draft_module.tokenizer = self.tokenizer
        self.target_layer_ids = list(draft_module.model.target_layer_ids)
        self.mask_token_id = int(draft_module.model.mask_token_id)

    @torch.inference_mode()
    def _prefill(self, prompt_ids, max_new_tokens, block_size, temperature):
        """Prefill target cache and seed the first generated token."""
        prefill_start = now(self.device)
        num_input_tokens = prompt_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens
        if self.mask_token_id is None:
            raise ValueError("mask_token_id is not set for block verifier.")
        output_ids = torch.full(
            (1, max_length + block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=self.device,
        )
        position_ids = torch.arange(output_ids.shape[1], device=self.device).unsqueeze(0)
        past_key_values_target = DynamicCache()
        with torch.no_grad():
            outputs = self.model(
                prompt_ids,
                position_ids=position_ids[:, :num_input_tokens],
                past_key_values=past_key_values_target,
                use_cache=True,
                logits_to_keep=1,
                output_hidden_states=block_size > 1,
            )
        output_ids[:, :num_input_tokens] = prompt_ids
        first_token = sample(outputs.logits, temperature)
        output_ids[:, num_input_tokens : num_input_tokens + 1] = first_token
        target_hidden = None
        if block_size > 1:
            target_hidden = extract_context_feature(outputs.hidden_states, self.target_layer_ids)
        return {
            "prompt_ids": prompt_ids,
            "num_input_tokens": num_input_tokens,
            "max_length": max_length,
            "block_size": block_size,
            "output_ids": output_ids,
            "position_ids": position_ids,
            "past_key_values_target": past_key_values_target,
            "target_hidden": target_hidden,
            "start": num_input_tokens,
            "prefill_ms": (now(self.device) - prefill_start) * 1000.0,
            "seed_token_id": int(first_token[0, 0].item()),
        }

    @torch.inference_mode()
    def _verify_block(self, state, block_output_ids):
        """Verify one drafted block with one target forward pass."""
        block_size = state["block_size"]
        start = state["start"]
        block_position_ids = state["position_ids"][:, start : start + block_size]
        verify_start = now(self.device)
        with torch.no_grad():
            output = self.model(
                block_output_ids,
                position_ids=block_position_ids,
                past_key_values=state["past_key_values_target"],
                use_cache=True,
                output_hidden_states=block_size > 1,
            )
        posterior = sample(output.logits, self.temperature)
        acceptance_length = int(
            (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        )
        state["output_ids"][:, start : start + acceptance_length + 1] = block_output_ids[
            :, : acceptance_length + 1
        ]
        state["output_ids"][:, start + acceptance_length + 1] = posterior[:, acceptance_length]
        state["start"] += acceptance_length + 1
        state["past_key_values_target"] = crop_past_key_values(
            state["past_key_values_target"],
            state["start"],
        )
        if block_size > 1:
            state["target_hidden"] = extract_context_feature(
                output.hidden_states,
                self.target_layer_ids,
            )[:, : acceptance_length + 1, :]

        bonus_token_id = int(posterior[0, acceptance_length].item())
        stop = False
        stop_reason = ""
        if self.stop_on_eos and bonus_token_id == self.tokenizer.eos_token_id:
            stop = True
            stop_reason = "eos"
        generated_tail = state["output_ids"][0, state["num_input_tokens"] : state["start"] + 1]
        if not stop and self.stop_on_eos and self.tokenizer.eos_token_id is not None:
            if torch.any(generated_tail == self.tokenizer.eos_token_id):
                stop = True
                stop_reason = "eos"
        if not stop and state["start"] >= state["max_length"]:
            stop = True
            stop_reason = "completion_limit"

        return {
            "accepted_length": acceptance_length,
            "bonus_token_id": bonus_token_id,
            "cloud_compute_ms": (now(self.device) - verify_start) * 1000.0,
            "stop": stop,
            "stop_reason": stop_reason,
        }

    def init_verify_state(self, request, prompt_ids, draft_session=None, shared_state=None):
        """Prefill target state and initialize block draft state."""
        draft_module = (shared_state or {}).get("_block_draft_module")
        if draft_module is None:
            raise RuntimeError("Block verifier requires a block draft module in shared state.")
        self._attach_target_resources(draft_module)
        state = self._prefill(
            prompt_ids=prompt_ids,
            max_new_tokens=request["completion_tokens"],
            block_size=self.draft_tokens_per_step,
            temperature=self.temperature,
        )
        draft_state = draft_module.build_draft_state(state)
        return {
            "verify_prefill_ms": state["prefill_ms"],
            "block_verify_state": state,
            "block_draft_state": draft_state,
            "block_draft_module": draft_module,
        }

    def verifier_prefill_options(self, init_payload, request, prompt_ids, draft_session=None, shared_state=None):
        """Declare prefill hidden-state transfer and block seed metadata."""
        del request, prompt_ids, draft_session, shared_state
        state = init_payload["block_verify_state"]
        prefill_payloads = []
        initial_hidden_bytes = 0
        if state.get("target_hidden") is not None:
            target_hidden = state["target_hidden"]
            initial_hidden_bytes = int(target_hidden.numel()) * int(target_hidden.element_size())
            prefill_payloads.append(
                tensor_payload(
                    target_hidden,
                    "cloud_to_edge",
                    "hidden_states",
                    "prefill_target_hidden",
                )
            )
        return {
            "prefill_ms": state["prefill_ms"],
            "include_draft_prefill": False,
            "prefill_payloads": prefill_payloads,
            "set_ttft_from_prefill": True,
            "seed_token_id": state["seed_token_id"],
            "seed_source": "verifier_seed",
            "prefill_network_extra": {"hidden_bytes": int(initial_hidden_bytes)},
            "extra_state": {
                "_block_verify_state": state,
                "_block_draft_state": init_payload["block_draft_state"],
            },
        }

    def should_return_empty_verify(self, draft_output, draft_ids):
        """Block terminal payload is keyed by `block_output_ids`, not draft ids."""
        del draft_ids
        return draft_output.get("block_output_ids") is None

    def empty_verify_payload(self, shared_state, draft_output):
        """Return a block-shaped terminal verifier payload."""
        payload = super().empty_verify_payload(shared_state, draft_output)
        payload["bonus_token_id"] = None
        return payload

    @specdec_verify
    def verify(self, session, *, draft_output, draft_ids):
        """Run one block verify round."""
        state = session["_shared_state"].get("_block_verify_state")
        if state is None:
            raise RuntimeError("Block shared state is missing verifier runtime state.")
        verify_payload = self._verify_block(
            state,
            draft_output["block_output_ids"].to(self.device),
        )
        accepted_length = int(verify_payload["accepted_length"])
        accepted_ids = list(draft_ids[:accepted_length])
        corrected_ids = [int(verify_payload["bonus_token_id"])]
        rejected_ids = list(draft_ids[accepted_length:])
        return VerifyResult(
            accepted_ids=accepted_ids,
            corrected_ids=corrected_ids,
            rejected_draft_ids=rejected_ids,
            payloads=[
                token_payload(draft_ids, "edge_to_cloud", "draft_ids"),
                custom_payload(
                    int(draft_output.get("downlink_hidden_bytes", 0) or 0),
                    "cloud_to_edge",
                    "target_hidden",
                ),
                control_payload("cloud_to_edge", "round_control"),
            ],
            cloud_compute_ms=float(verify_payload.get("cloud_compute_ms", 0.0) or 0.0),
            stop=bool(verify_payload.get("stop", False)),
            stop_reason=str(verify_payload.get("stop_reason", "") or ""),
            token_provenance=[
                *({"token_id": int(token_id), "source": "draft_accepted"} for token_id in accepted_ids),
                {"token_id": int(verify_payload["bonus_token_id"]), "source": "verifier_bonus"},
            ],
            round_stats={
                "draft_count": len(draft_ids),
                "accepted_length": accepted_length,
                "corrected_count": len(corrected_ids),
                "rejected_draft_count": max(len(draft_ids) - len(accepted_ids), 0),
                "rejected_draft_ids": list(rejected_ids),
                "accepted_ids": list(accepted_ids),
                "corrected_ids": list(corrected_ids),
                "stop_reason": str(verify_payload.get("stop_reason", "") or ""),
                "cloud_compute_ms": float(verify_payload.get("cloud_compute_ms", 0.0) or 0.0),
                "edge_compute_ms": float(draft_output.get("edge_compute_ms", 0.0) or 0.0),
                "bonus_token_id": int(verify_payload["bonus_token_id"]),
            },
            extra_fields={"bonus_token_id": int(verify_payload["bonus_token_id"])},
        )

    def after_verify(self, session, draft_output, verify_payload, response):
        """Refresh draft-side block state from target verifier state."""
        del draft_output, verify_payload, response
        shared_state = session["_shared_state"]
        draft_module = shared_state.get("_block_draft_module")
        draft_state = shared_state.get("_block_draft_state")
        verify_state = shared_state.get("_block_verify_state")
        if draft_module is not None and draft_state is not None and verify_state is not None:
            draft_module.sync_draft_state(draft_state, verify_state)

    def encode_cloud_prompt(self, request):
        """Encode a cloud-only block prompt."""
        prompt_ids = self.encode_prompt(request)
        return {
            "prompt_ids": prompt_ids,
            "transport_payload": token_payload(prompt_ids, "edge_to_cloud", "prompt_ids"),
            "completion_payload": lambda completion_ids: token_payload(
                completion_ids,
                "cloud_to_edge",
                "completion_ids",
            ),
        }

    def cloud_generate(self, request, prompt_payload):
        """Generate one cloud-only block-path completion."""
        if self.block_cloud_only_backend == "transformers":
            return self._cloud_generate_transformers(prompt_payload["prompt_ids"], request["completion_tokens"])
        return self._cloud_generate_custom(prompt_payload["prompt_ids"], request["completion_tokens"])

    @torch.inference_mode()
    def _cloud_generate_custom(self, prompt_ids, max_new_tokens):
        """Run target-only autoregressive decoding with explicit stepping."""
        generated = []
        token_timestamps_ms = []
        start = now(self.device)
        ttft_ms = None
        with torch.no_grad():
            outputs = self.model(input_ids=prompt_ids, use_cache=True)
            prefill_ms = (now(self.device) - start) * 1000.0
            cache = outputs.past_key_values
            next_token = sample(outputs.logits[:, -1:, :], self.temperature)
            token_id = int(next_token[0, 0].item())
            for _ in range(max_new_tokens):
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
                next_token = sample(outputs.logits[:, -1:, :], self.temperature)
                token_id = int(next_token[0, 0].item())
        total_ms = (now(self.device) - start) * 1000.0
        return {
            "completion_ids": generated,
            "prefill_ms": prefill_ms if max_new_tokens > 0 else total_ms,
            "ttft_ms": ttft_ms or total_ms,
            "total_ms": total_ms,
            "token_timestamps_ms": token_timestamps_ms,
            "stop_reason": "eos" if generated and generated[-1] == self.tokenizer.eos_token_id else "completion_limit",
        }

    @torch.inference_mode()
    def _cloud_generate_transformers(self, prompt_ids, max_new_tokens):
        """Run target-only autoregressive decoding with `transformers.generate()`."""
        start = now(self.device)
        attention_mask = torch.ones_like(prompt_ids, device=prompt_ids.device)
        streamer = _TokenTimingStreamer(prompt_ids.shape[1], start, self.device)
        do_sample = float(self.temperature or 0.0) >= 1e-5
        generate_kwargs = {
            "input_ids": prompt_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": int(max_new_tokens),
            "use_cache": True,
            "return_dict_in_generate": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id if self.stop_on_eos else None,
            "streamer": streamer,
        }
        if do_sample:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = float(self.temperature)
        else:
            generate_kwargs["do_sample"] = False

        with torch.no_grad():
            output = self.model.generate(**generate_kwargs)

        total_ms = (now(self.device) - start) * 1000.0
        sequence = output.sequences[0]
        completion_ids = sequence[prompt_ids.shape[1] :].tolist()
        token_timestamps_ms = streamer.token_timestamps_ms[: len(completion_ids)]
        if len(token_timestamps_ms) < len(completion_ids):
            token_timestamps_ms.extend([total_ms] * (len(completion_ids) - len(token_timestamps_ms)))
        ttft_ms = token_timestamps_ms[0] if token_timestamps_ms else total_ms
        return {
            "completion_ids": completion_ids,
            "prefill_ms": ttft_ms,
            "ttft_ms": ttft_ms,
            "total_ms": total_ms,
            "token_timestamps_ms": token_timestamps_ms,
            "stop_reason": "eos" if completion_ids and completion_ids[-1] == self.tokenizer.eos_token_id else "completion_limit",
        }

    @staticmethod
    def cloud_extra_fields(*, generation, total_ms, network):
        """Build block cloud-only timing fields."""
        decode_compute_ms = max(total_ms - generation["prefill_ms"] - network["network_ms"], 0.0)
        return {
            "time_breakdown": {
                "prefill_compute_ms": round(float(generation["prefill_ms"]), 6),
                "decode_compute_ms": round(decode_compute_ms, 6),
                "edge_decode_compute_ms": 0.0,
                "cloud_verify_compute_ms": round(decode_compute_ms, 6),
                "network_ms": round(float(network["network_ms"]), 6),
                "network_propagation_ms": round(float(network.get("propagation_ms", 0.0)), 6),
                "network_transfer_ms": round(float(network.get("transfer_ms", 0.0)), 6),
            }
        }
