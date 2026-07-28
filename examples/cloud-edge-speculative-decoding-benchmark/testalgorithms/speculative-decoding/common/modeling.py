"""KV-cache helpers shared by speculative decoding implementations."""

from __future__ import annotations

import torch


def _slice_cache_tensor(tensor, end_pos, *, is_key=False):
    """Slice a cache tensor on its sequence dimension."""
    if tensor.dim() == 4:
        return tensor[:, :, :end_pos, :]
    if is_key:
        return tensor[:, :, :end_pos]
    return tensor[:, :end_pos, :]


def _crop_legacy_cache(past_key_values, end_pos):
    """Crop tuple/list-style past_key_values."""
    trimmed = []
    for layer in past_key_values:
        key = _slice_cache_tensor(layer[0], end_pos, is_key=True)
        value = _slice_cache_tensor(layer[1], end_pos)
        trimmed.append((key, value, *layer[2:]))
    return tuple(trimmed)


def _crop_dynamic_cache_fields(past_key_values, end_pos):
    """Crop common DynamicCache tensor fields in place."""
    updated = False
    for attr_name, is_key in (("key_cache", True), ("value_cache", False)):
        cache = getattr(past_key_values, attr_name, None)
        if cache is None:
            continue
        for index, tensor in enumerate(cache):
            if tensor is not None:
                cache[index] = _slice_cache_tensor(tensor, end_pos, is_key=is_key)
                updated = True
    return updated


def _crop_layer_objects(past_key_values, end_pos):
    """Crop layer-based cache implementations in place when exposed."""
    layers = getattr(past_key_values, "layers", None)
    if layers is None:
        return False

    updated = False
    for layer in layers:
        for attr_name in ("keys", "values", "key_cache", "value_cache"):
            tensor = getattr(layer, attr_name, None)
            if tensor is not None:
                setattr(layer, attr_name, _slice_cache_tensor(
                    tensor,
                    end_pos,
                    is_key=attr_name in {"keys", "key_cache"},
                ))
                updated = True
    return updated


def crop_past_key_values(past_key_values, end_pos):
    """Trim a Hugging Face KV cache to the given logical token length.

    Different transformers releases expose different cache APIs. This helper
    prefers the native cache method when available, then falls back to common
    DynamicCache fields and finally tuple-style legacy caches.
    """
    if past_key_values is None:
        return None

    end_pos = int(end_pos)
    crop = getattr(past_key_values, "crop", None)
    if callable(crop):
        crop(end_pos)
        return past_key_values

    if _crop_dynamic_cache_fields(past_key_values, end_pos):
        return past_key_values

    if _crop_layer_objects(past_key_values, end_pos):
        return past_key_values

    if isinstance(past_key_values, (list, tuple)):
        return _crop_legacy_cache(past_key_values, end_pos)

    raise TypeError(
        f"Unsupported past_key_values type for cache cropping: {type(past_key_values)!r}"
    )

def sample_token_id(logits, temperature):
    """Sample one token id from logits."""
    if float(temperature or 0.0) < 1e-5:
        return int(torch.argmax(logits, dim=-1).reshape(-1)[0].item())
    scaled = logits.float() / max(float(temperature or 0.0), 1e-5)
    return int(torch.distributions.Categorical(logits=scaled).sample().reshape(-1)[0].item())


def scaled_logits(logits, temperature):
    """Apply temperature scaling to logits."""
    return logits.float() / max(float(temperature or 0.0), 1e-5)


def build_topk_distribution(logits, temperature, top_k):
    """Build a sparse top-k probability payload from one logits tensor."""
    scaled = scaled_logits(logits, temperature)
    vocab_size = int(scaled.shape[-1])
    top_k = max(1, min(int(top_k or 0), vocab_size))
    topk_logits, topk_token_ids = torch.topk(scaled, k=top_k, dim=-1)
    topk_token_probs = torch.softmax(topk_logits, dim=-1)
    return {
        "representation": "topk_probs",
        "top_k": int(top_k),
        "vocab_size": int(vocab_size),
        "token_ids": topk_token_ids.detach().cpu(),
        "token_probs": topk_token_probs.detach().cpu(),
    }


def sample_token_id_from_topk(distribution):
    """Sample one token id from a sparse top-k payload."""
    token_ids = distribution["token_ids"].reshape(-1)
    token_probs = distribution["token_probs"].reshape(-1)
    sampled_index = int(torch.distributions.Categorical(probs=token_probs).sample().item())
    return int(token_ids[sampled_index].item())


def greedy_token_id(logits):
    """Return the argmax token id."""
    return int(torch.argmax(logits, dim=-1).reshape(-1)[0].item())


def probs_from_logits(logits, temperature):
    """Convert logits into a normalized probability vector."""
    scaled = logits.float() / max(float(temperature or 0.0), 1e-5)
    return torch.softmax(scaled, dim=-1).reshape(-1)


def sample_from_probs(probs):
    """Sample one token id from probabilities."""
    return int(torch.distributions.Categorical(probs=probs).sample().item())


def sample_from_residual(p_probs, q_probs):
    """Sample a correction token from max(p - q, 0)."""
    residual = torch.clamp(p_probs - q_probs, min=0.0)
    if float(residual.sum().item()) <= 0.0:
        return sample_from_probs(p_probs)
    residual = residual / residual.sum()
    return sample_from_probs(residual)


def is_topk_probability_payload(payload):
    """Return whether a draft distribution uses sparse top-k encoding."""
    return isinstance(payload, dict) and payload.get("representation") == "topk_probs"


def lookup_topk_probability(token_id, payload, device):
    """Look up one token probability from a sparse top-k payload."""
    token_ids = payload["token_ids"].reshape(-1).to(device)
    token_probs = payload["token_probs"].reshape(-1).to(device=device, dtype=torch.float32)
    matches = torch.nonzero(token_ids == int(token_id), as_tuple=False)
    if matches.numel() == 0:
        raise RuntimeError(f"Draft token {token_id} is not present in transported top-k support.")
    return float(token_probs[int(matches[0].item())].item())


def sample_from_sparse_residual(p_probs, payload):
    """Sample a correction token from max(p - q_topk, 0)."""
    token_ids = payload["token_ids"].reshape(-1).to(p_probs.device)
    token_probs = payload["token_probs"].reshape(-1).to(device=p_probs.device, dtype=p_probs.dtype)
    residual = p_probs.clone()
    residual[token_ids] = torch.clamp(residual[token_ids] - token_probs, min=0.0)
    if float(residual.sum().item()) <= 0.0:
        return sample_from_probs(p_probs)
    residual = residual / residual.sum()
    return sample_from_probs(residual)
