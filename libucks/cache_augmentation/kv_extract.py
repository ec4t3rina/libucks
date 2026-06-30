"""KV cache extraction from a frozen receiver over bucket source text.

Pipeline: text → tokenize → frozen forward with use_cache=True → captured
past_key_values (DynamicCache in transformers v5.x with .layers list).

We serialise as a flat tensor dict per bucket:
    {
        "layer_<i>_K": (1, n_kv_heads, T, head_dim) tensor,
        "layer_<i>_V": (1, n_kv_heads, T, head_dim) tensor,
    }

This is what BucketKVCache.save persists; reconstruction back to a
DynamicCache happens in the augmented-decode path (Phase 4-C.4).
"""
from __future__ import annotations

import sys
from typing import Any

import torch


def _log(msg: str) -> None:
    print(f"[libucks:cache_aug] {msg}", file=sys.stderr, flush=True)


@torch.no_grad()
def extract_bucket_kv(
    model: torch.nn.Module,
    tokenizer: Any,
    bucket_text: str,
    *,
    max_tokens: int = 1024,
) -> dict[str, torch.Tensor]:
    """Run the frozen receiver forward over bucket text; return a flat tensor
    dict of per-layer K and V suitable for safetensors serialisation.

    Args:
        model: a frozen HF causal LM (Qwen 2.5-3B in libucks's case).
        tokenizer: the matching tokenizer.
        bucket_text: text content to encode (typically `_collect_source_text`
            output for the bucket — the whole bucket, not positionally
            truncated, since cache aug is meant to consume the full bucket
            via cross-attention).
        max_tokens: truncate inputs at this many tokens to bound storage.
            1024 is conservative for typical buckets; large buckets can be
            re-extracted at higher caps later.

    Returns:
        {"layer_0_K": ..., "layer_0_V": ..., ..., "layer_<L-1>_V": ...,
         "_meta_seq_len": scalar tensor, "_meta_n_layers": scalar tensor}.
        Tensors are detached, on CPU, in the model's compute dtype.
    """
    device = next(model.parameters()).device
    enc = tokenizer(
        bucket_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    pkv = out.past_key_values  # DynamicCache in transformers v5.x

    if not hasattr(pkv, "layers"):
        raise RuntimeError(
            f"Unsupported past_key_values type {type(pkv).__name__}; "
            "expected DynamicCache with .layers attribute (transformers v5.x)."
        )

    flat: dict[str, torch.Tensor] = {}
    for i, layer in enumerate(pkv.layers):
        # Detach + move to CPU before storage. Keep compute dtype (bf16
        # typically); we'll quantise to int8 in a follow-up if storage
        # is the bottleneck.
        flat[f"layer_{i}_K"] = layer.keys.detach().cpu().contiguous()
        flat[f"layer_{i}_V"] = layer.values.detach().cpu().contiguous()

    seq_len = int(input_ids.shape[1])
    flat["_meta_seq_len"] = torch.tensor([seq_len], dtype=torch.int32)
    flat["_meta_n_layers"] = torch.tensor([len(pkv.layers)], dtype=torch.int32)
    _log(
        f"extract: seq_len={seq_len} layers={len(pkv.layers)} "
        f"per_layer_K_shape={tuple(pkv.layers[0].keys.shape)} "
        f"dtype={pkv.layers[0].keys.dtype}"
    )
    return flat


def restore_dynamic_cache(flat: dict[str, torch.Tensor], device: torch.device) -> Any:
    """Inverse of extract_bucket_kv: rebuild a DynamicCache from the flat dict.

    Returns a transformers.cache_utils.DynamicCache that can be passed via
    past_key_values= to a subsequent model.generate or model.forward call.
    """
    from transformers.cache_utils import DynamicCache

    n_layers = int(flat["_meta_n_layers"].item())
    cache = DynamicCache()
    # DynamicCache.update(key_states, value_states, layer_idx) is the
    # documented API for filling layers; it auto-initialises a new
    # DynamicLayer at the given index.
    for i in range(n_layers):
        k = flat[f"layer_{i}_K"].to(device)
        v = flat[f"layer_{i}_V"].to(device)
        cache.update(k, v, layer_idx=i)
    return cache
