"""Augmented-cache decode — the per-query inference path.

Per-query flow (when cache_aug is active):

  1. For each routed bucket: load BucketKVCache(b) from disk → coprocessor(cache_b)
     → z_b ∈ ℝ^(1, K=64, 2048)
  2. CrossBucketFusion([z_1, ..., z_k]) → z_fused ∈ ℝ^(1, K=64, 2048)
  3. Frozen receiver forward over `verbatim + query + asst_cue` (text) → C_input
     (the standard prompt KV cache, no augmentation)
  4. Frozen receiver forward over z_fused (as inputs_embeds, length K) → C_z
     (the "augmentation" KV cache, the slot tokens' contribution to attention)
  5. Concatenate C_input + C_z layer-by-layer into one augmented DynamicCache.
     The asst_cue token is the LAST token in C_input, so generation begins from
     that position; z_fused is APPENDED, so the receiver sees [text..., z...]
     when computing the next-token logits.
  6. model.generate(input_ids=<new_token_seed>, past_key_values=C_aug, ...).

Architectural rule: this module is the only place that constructs an augmented
cache. It is called exclusively from Translator.synthesize_cache_aug() so the
"Translator is the sole decode point" constraint is preserved.

NOTE — augmentation order: DeepMind appends z AFTER the input; we follow that.
Some ablations may want z prepended (so attention's causal direction puts the
input as "what z is about"). We keep the appended variant as default; the
prepended variant is a 4-C.6 ablation if needed.
"""
from __future__ import annotations

import sys
from typing import List, Optional

import torch

from libucks.cache_augmentation.bucket_kv_cache import BucketKVCache
from libucks.cache_augmentation.coprocessor import Coprocessor
from libucks.cache_augmentation.fusion import CrossBucketFusion


def _log(msg: str) -> None:
    print(f"[libucks:cache_aug] {msg}", file=sys.stderr, flush=True)


@torch.no_grad()
def build_z_fused(
    coprocessor: Coprocessor,
    fusion: CrossBucketFusion,
    bucket_kv_cache: BucketKVCache,
    bucket_ids: List[str],
    bucket_chunks: dict[str, list],
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Build z_fused for the routed buckets.

    Args:
        coprocessor: the per-bucket Coprocessor.
        fusion: the multi-bucket fusion block.
        bucket_kv_cache: disk-backed cache; we read each bucket's flat KV from it.
        bucket_ids: top-k routed bucket ids (ordered).
        bucket_chunks: mapping bucket_id → ChunkMetadata list, used for
            cache freshness check (signature comparison).
        device: target device for the fusion forward.

    Returns:
        z_fused ∈ ℝ^(1, K, hidden_dim) on `device`, or None if no bucket
        has a fresh cache available (caller should fall back).
    """
    bucket_z: List[torch.Tensor] = []
    for bid in bucket_ids:
        chunks = bucket_chunks.get(bid, [])
        flat = bucket_kv_cache.load(bid, chunks, device="cpu")
        if flat is None:
            _log(f"build_z_fused: bucket={bid[:10]} cache miss; skipping")
            continue
        z_b = coprocessor(flat)            # (1, K, hidden_dim)
        bucket_z.append(z_b.to(device))

    if not bucket_z:
        _log("build_z_fused: no fresh caches; returning None")
        return None

    z_fused = fusion(bucket_z)              # (1, K, hidden_dim)
    return z_fused


@torch.no_grad()
def augmented_decode(
    base_model,
    tokenizer,
    z_fused: torch.Tensor,
    query: str,
    verbatim: str = "",
    *,
    max_new_tokens: int = 120,
    asst_cue: str = " The answer:",
    # Phase 4-C.6 salvage: Cold Stop entropy gate (Soft Thinking §3.3 adapted).
    # When the receiver's logit entropy stays above `cold_stop_entropy` for
    # `cold_stop_consecutive` consecutive generated tokens, stop. The cache-aug
    # failure mode is HIGH-entropy random vocab fragments; this cuts the garbage
    # tail while keeping the (rare) topical prefix. Set entropy threshold to
    # `None` to disable. Threshold ~4.0 ≈ uniform over ~55 tokens.
    cold_stop_entropy: float | None = 4.0,
    cold_stop_consecutive: int = 3,
) -> str:
    """Run the receiver decode with z_fused appended into its KV cache.

    Args:
        base_model: frozen Qwen 2.5-3B.
        tokenizer: matching tokenizer.
        z_fused: (1, K, hidden_dim) the fused soft prompt to inject.
        query: the user query (text).
        verbatim: optional source code prepended to the prompt for hybrid mode.
        max_new_tokens: generation budget.
        asst_cue: a short "answer cue" text appended after the query before
            generation, mirroring the existing Translator's framing.

    Returns:
        Decoded text (continuation only — prompt tokens stripped).
    """
    device = next(base_model.parameters()).device
    receiver_dtype = next(base_model.parameters()).dtype

    # ------------------------------------------------------------------
    # Stage 1: build C_input — the standard prompt KV (no augmentation).
    # ------------------------------------------------------------------
    prompt_parts = []
    if verbatim:
        prompt_parts.append(verbatim.strip())
    prompt_parts.append(f"Question: {query.strip()}")
    prompt_parts.append(asst_cue.strip())
    prompt = "\n\n".join(prompt_parts)

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=3500,
    ).to(device)
    input_ids = enc["input_ids"]

    out = base_model(
        input_ids=input_ids,
        use_cache=True,
    )
    cache_input = out.past_key_values  # DynamicCache

    # ------------------------------------------------------------------
    # Stage 2: build C_z — the augmentation KV (from z_fused embeddings).
    # ------------------------------------------------------------------
    # Cast z_fused to receiver dtype + device.
    z = z_fused.to(device=device, dtype=receiver_dtype)
    # Run a separate forward over z as inputs_embeds to derive its KV.
    # We do NOT chain past_key_values=cache_input here on purpose — we want
    # the z forward to be conditioned on its own causal context only; appending
    # to the input cache happens after.
    out_z = base_model(
        inputs_embeds=z,
        use_cache=True,
    )
    cache_z = out_z.past_key_values

    # ------------------------------------------------------------------
    # Stage 3: concatenate cache_input + cache_z layer-by-layer.
    # ------------------------------------------------------------------
    from transformers.cache_utils import DynamicCache

    aug_cache = DynamicCache()
    n_layers = len(cache_input.layers)
    for i in range(n_layers):
        k_in = cache_input.layers[i].keys
        v_in = cache_input.layers[i].values
        k_z = cache_z.layers[i].keys
        v_z = cache_z.layers[i].values
        # Concat along the seq_len dim (dim=2): (B, H_kv, T_in + T_z, D)
        k_cat = torch.cat([k_in, k_z], dim=2)
        v_cat = torch.cat([v_in, v_z], dim=2)
        aug_cache.update(k_cat, v_cat, layer_idx=i)

    # ------------------------------------------------------------------
    # Stage 4: manual greedy decode with optional Cold Stop entropy gate.
    # ------------------------------------------------------------------
    # We replace model.generate with a per-token loop so we can monitor
    # logit entropy at each step (Soft Thinking §3.3 Cold Stop adapted).
    # Seed: feed the last input token; subsequent tokens are appended via
    # past_key_values updates returned by each forward.
    import torch.nn.functional as F

    cache = aug_cache
    current_token = input_ids[:, -1:].to(device)  # (1, 1)
    generated_ids: list[int] = []
    eos_id = tokenizer.eos_token_id
    high_entropy_streak = 0
    cold_stop_triggered = False

    for step in range(max_new_tokens):
        out = base_model(
            input_ids=current_token,
            past_key_values=cache,
            use_cache=True,
        )
        cache = out.past_key_values
        logits = out.logits[0, -1, :].float()  # (V,)

        # Cold Stop: per-token entropy on the next-token distribution.
        if cold_stop_entropy is not None:
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum().item()
            if entropy > cold_stop_entropy:
                high_entropy_streak += 1
                if high_entropy_streak >= cold_stop_consecutive:
                    cold_stop_triggered = True
                    break
            else:
                high_entropy_streak = 0

        next_id = int(logits.argmax().item())
        if next_id == eos_id:
            break
        generated_ids.append(next_id)
        current_token = torch.tensor([[next_id]], dtype=torch.long, device=device)

    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    _log(
        f"augmented_decode: {len(generated_ids)} new tokens, {len(text)} chars"
        + (f" (cold-stopped at step {step})" if cold_stop_triggered else "")
    )
    return text
