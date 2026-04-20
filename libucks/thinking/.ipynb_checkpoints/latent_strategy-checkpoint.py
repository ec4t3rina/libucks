"""LatentStrategy — V2 implementation using local HuggingFace transformers.

Librarians call encode() and reason() — both return torch.Tensor hidden states
from the model's last layer. No text is generated at these call sites.

ONLY the Translator calls decode() to produce natural-language output by
projecting hidden states through the LM head and running autoregressive generation.

See ARCHITECTURE.md §4 for the architectural constraints.
"""
from __future__ import annotations

import sys

from libucks.thinking.base import Representation, ThinkingStrategy


_ANCHOR_PROMPT = "<|im_start|>assistant\n"


def _log(msg: str) -> None:
    print(f"[libucks:latent] {msg}", file=sys.stderr, flush=True)


class LatentStrategy(ThinkingStrategy):
    def __init__(
        self,
        model_manager: object | None = None,
        compressor: object | None = None,
        injection_gate: float = 0.3,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.3,
        receive_temperature: float = 0.6,
        receive_top_p: float = 0.9,
        receive_top_k: int = 50,
        receive_repetition_penalty: float = 1.2,
    ) -> None:
        self._mgr = model_manager
        self._compressor = compressor
        # Residual injection gate g: x_soft = dummy + g*(hidden_matched - dummy).
        # Starting at 0.1 keeps 90% of the output on the native embedding manifold
        # and injects only 10% adapter signal, preventing generation collapse from
        # off-manifold perturbations.
        self._injection_gate = injection_gate
        # Sampling parameters to break greedy (argmax) attractor loops.
        # Argmax at T=0 creates deterministic 3-token cycles ("1. 1. 1.") because
        # the soft-prompt primes a peaked distribution.  Multinomial sampling with
        # temperature > 0 breaks the cycle; repetition_penalty further suppresses
        # tokens already in the generated sequence.
        self._temperature = temperature
        self._top_p = top_p
        self._repetition_penalty = repetition_penalty
        # Separate sampling params for receive() — the Base model (not Instruct)
        # has a much flatter logit distribution. top_k=50 is applied first as a
        # hard candidate cap before top_p, preventing tail bleed on a 150k vocab.
        # repetition_penalty=1.2 breaks degenerate loops (~2 cycles) without
        # pushing probability mass into multilingual tokens (which 1.3+ does).
        self._receive_temperature = receive_temperature
        self._receive_top_p = receive_top_p
        self._receive_top_k = receive_top_k
        self._receive_repetition_penalty = receive_repetition_penalty
        # MPS has a single Metal command queue. Concurrent submissions from
        # multiple asyncio coroutines (e.g. 3 Librarians via asyncio.gather)
        # deadlock against each other. This lock serialises all device inference.
        import asyncio
        self._device_lock = asyncio.Lock()

    @property
    def hidden_dim(self) -> int:
        """Return the hidden_size of the encoder model."""
        return self._mgr.hidden_dim

    def _sample_next_token(
        self,
        logits: "torch.Tensor",
        generated_ids: list[int],
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
    ) -> "torch.Tensor":
        """Sample the next token with repetition penalty, temperature, top-k, and top-p.

        top_k is applied before top_p: it first hard-caps the candidate set to
        the k highest-logit tokens, then top_p further narrows within that set.
        This prevents tail bleed on flat 150k-vocab Base model distributions where
        top_p alone can include thousands of candidates.

        Args:
            logits: shape (vocab_size,) — raw logits for the next-token position.
            generated_ids: token IDs produced so far (used for repetition penalty).

        Returns:
            torch.Tensor of shape (1,) — the sampled token ID.
        """
        import torch
        import torch.nn.functional as F

        _rep_penalty = repetition_penalty if repetition_penalty is not None else self._repetition_penalty
        _temperature = temperature if temperature is not None else self._temperature
        _top_p = top_p if top_p is not None else self._top_p
        _top_k = top_k  # None means no top-k filtering

        logits = logits.float().clone()

        # Repetition penalty: HuggingFace convention —
        #   positive logit → divide by penalty (reduces)
        #   negative logit → multiply by penalty (makes more negative)
        if _rep_penalty != 1.0 and generated_ids:
            for token_id in set(generated_ids):
                if logits[token_id] > 0:
                    logits[token_id] /= _rep_penalty
                else:
                    logits[token_id] *= _rep_penalty

        # Temperature scaling
        logits = logits / max(_temperature, 1e-8)

        # Top-k filtering — zero out everything outside the k highest logits.
        # Applied before top-p so nucleus filtering operates on a bounded set.
        if _top_k is not None and _top_k > 0:
            top_k_vals = torch.topk(logits, min(_top_k, logits.size(-1)))[0]
            logits[logits < top_k_vals[-1]] = float("-inf")

        # Top-p (nucleus) filtering — zero out the tail below the probability mass
        if _top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Keep the first token that pushes cumulative mass above top_p;
            # remove everything after it.
            sorted_indices_to_remove = cumulative_probs > _top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False   # always keep the top token
            logits[sorted_indices[sorted_indices_to_remove]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    async def encode(self, text: str) -> "torch.Tensor":
        """Encode text into last-layer hidden states via a single forward pass.

        Returns:
            torch.Tensor of shape (seq_len, hidden_dim).
        """
        import torch

        model = self._mgr.get_model()
        tokenizer = self._mgr.get_tokenizer()
        device = self._mgr.device

        async with self._device_lock:
            # no_grad (not inference_mode): prevents gradient computation in the
            # encoder while keeping returned tensors as normal (non-inference)
            # tensors.  inference_mode marks outputs as inference tensors, which
            # cannot be saved for backward when they later flow into the adapter
            # forward pass during LoRA receiver training.
            with torch.no_grad():
                inputs = tokenizer(text, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                output = model(**inputs, output_hidden_states=True)
                # hidden_states[-1]: (1, seq_len, hidden_dim) → (seq_len, hidden_dim)
                hidden = output.hidden_states[-1].squeeze(0).contiguous()
                del output  # release full 36-layer hidden_states tuple
                return hidden

    async def reason(self, query: str, context: str) -> "torch.Tensor":
        """Produce a hidden-state Representation for a query given context.

        Constructs the standard prompt template, runs a single forward pass,
        and returns the last-layer hidden states. model.generate() is never
        called here — only the Translator is permitted to decode.

        Returns:
            torch.Tensor of shape (seq_len, hidden_dim).
        """
        import torch

        model = self._mgr.get_model()
        tokenizer = self._mgr.get_tokenizer()
        device = self._mgr.device

        prompt = f"{context}\n\n{query}"

        _log(f"reason: tokenizing ({len(prompt)} chars, device={device})")
        async with self._device_lock:
            with torch.no_grad():  # see encode() comment: no_grad not inference_mode
                # Truncate to 256 tokens — Qwen forward-pass latency on MPS
                # scales with sequence length; 485-token prompts push 3 serial
                # Librarian calls past the 60-second MCP timeout.
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=256,
                    truncation=True,
                )
                seq_len = inputs["input_ids"].shape[-1]
                inputs = {k: v.to(device) for k, v in inputs.items()}
                _log(f"reason: forward pass (seq_len={seq_len})")
                output = model(**inputs, output_hidden_states=True)
                hidden = output.hidden_states[-1].squeeze(0).contiguous()
                del output  # release full 36-layer hidden_states tuple
                _log(f"reason: forward pass complete, hidden={tuple(hidden.shape)}")

            if self._compressor is not None:
                _log(f"reason: compressing ({hidden.shape[0]} → {self._compressor.compression_steps} steps)")
                with torch.inference_mode():
                    hidden = self._compressor(hidden).contiguous()
                _log(f"reason: compressed to {tuple(hidden.shape)}")

        return hidden

    async def decode(self, result: "torch.Tensor") -> str:
        """Convert hidden-state tensor to natural language via Residual Anchoring.

        Three-stage injection (Vision Wormhole, Eq. 2 + Eq. 9):

        1. NormMatch (Eq. 9): rescale adapter output per-row to match the
           embedding layer's mean norm — places hidden_matched at embed_scale.

        2. Residual Anchoring (Eq. 2): blend with a dummy baseline of K EOS-token
           embeddings that are guaranteed to be on Qwen's native manifold:
               x_soft = dummy_embeds + g * (hidden_matched - dummy_embeds)
           At g=0.1 (default), 90% of the signal comes from the real embedding
           manifold and only 10% from the adapter output.  A final re-normalisation
           to embed_scale enforces the manifold norm constraint regardless of g.

        3. Position ID Alignment: pass explicit position_ids on every model() call
           so RoPE offsets are correct for the KV cache even when inputs_embeds is
           used instead of input_ids on the first call.

        This is the ONLY authorised call site for generative inference in the system.

        Args:
            result: torch.Tensor of shape (seq_len, hidden_dim) or
                    (1, seq_len, hidden_dim).

        Returns:
            Decoded natural-language string.
        """
        import torch
        from transformers import DynamicCache

        model = self._mgr.get_model()
        tokenizer = self._mgr.get_tokenizer()
        device = self._mgr.device

        _log(f"decode: received tensor {tuple(result.shape)}, device={device}")

        async with self._device_lock:
            with torch.no_grad():
                # --- Normalise input shape ---
                hidden = result.to(device)
                if hidden.dim() == 2:
                    hidden = hidden.unsqueeze(0)   # (1, K, d)
                hidden = hidden.contiguous()       # MPS: avoids silent hang on non-contiguous strides
                K = hidden.shape[1]

                # --- Step 1: NormMatch (Vision Wormhole, Eq. 9) ---
                # Rescale each adapter output row to embed_scale so hidden_matched
                # and the dummy baseline are at the same magnitude.
                embed_scale = model.model.embed_tokens.weight.norm(dim=-1).mean()
                adapter_norms = hidden.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                hidden_matched = hidden * (embed_scale / adapter_norms)  # (1, K, d)
                _log(f"decode: isnan={hidden_matched.isnan().any().item()}, isinf={hidden_matched.isinf().any().item()}")
                _log(
                    f"decode: NormMatch embed_scale={embed_scale.item():.3f}, "
                    f"mean_adapter_norm={adapter_norms.mean().item():.3f}"
                )
                

                # --- Step 2: Residual Anchoring (Vision Wormhole, Eq. 2) ---
                # Embed K copies of eos_token_id as the dummy baseline.  EOS is always
                # available and its embedding sits firmly on the native manifold.
                space_id = int(tokenizer(" ", return_tensors="pt", add_special_tokens=False)["input_ids"].flatten()[-1].item())
                dummy_ids = torch.full(
                    (1, K), space_id, dtype=torch.long, device=device
                )
                dummy_embeds = model.model.embed_tokens(dummy_ids)   # (1, K, d)

                # Blend: x_soft = dummy + gate * (hidden_matched - dummy)
                delta = hidden_matched - dummy_embeds
                x_soft = dummy_embeds + self._injection_gate * delta  # (1, K, d)

                # Re-normalise to embed_scale: enforces manifold norm constraint after
                # blending (the convex combination of two unit-scale vectors has norm
                # ≈ 0.9 × embed_scale at gate=0.1 for orthogonal random vectors).
                x_soft_norms = x_soft.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                x_soft = x_soft * (embed_scale / x_soft_norms)
                _log(
                    f"decode: residual gate={self._injection_gate}, "
                    f"mean_delta_norm={delta.norm(dim=-1).mean().item():.3f}"
                )

                # --- Step 3: Embed anchor prompt for semantic texture ---
                anchor_ids = tokenizer(
                    _ANCHOR_PROMPT, return_tensors="pt", add_special_tokens=False
                )["input_ids"].to(device)                              # (1, L)
                anchor_embeds = model.model.embed_tokens(anchor_ids)   # (1, L, d)

                # --- Step 4: Concatenate and build position IDs ---
                inputs_embeds = torch.cat(
                    [x_soft, anchor_embeds], dim=1
                )  # (1, K+L, d)
                prefix_len = inputs_embeds.shape[1]

                # Explicit position_ids ensure RoPE is applied at the correct offsets
                # even when the model receives inputs_embeds instead of input_ids.
                # Without this, some Qwen2.5 versions revert to position 0 for the
                # first token of a DynamicCache continuation.
                prefix_pos = torch.arange(
                    prefix_len, dtype=torch.long, device=device
                ).unsqueeze(0)  # (1, K+L)
                # Cast to the model's dtype (float16 on CUDA/MPS) so the lm_head
                # matmul doesn't hit a Half/Float mismatch from the float32
                # adapter output flowing through the embedding concatenation.
                inputs_embeds = inputs_embeds.to(model.dtype)
                _log(f"decode: inputs_embeds shape={tuple(inputs_embeds.shape)}, dtype={inputs_embeds.dtype}")

                # --- Step 5: Prefix pass ---
                #
                # WHY NOT model.generate():
                # model.generate() calls _get_cache() which constructs a StaticCache
                # regardless of past_key_values or generation_config patches. For
                # Qwen2.5-3B on MPS:
                #   36 layers × 2 × 8 kv_heads × 32768 × 128 × float16 ≈ 4.8 GB
                # Metal rejects any single NDArray > 2^32 bytes — instant SIGABRT.
                #
                # DynamicCache grows ~9 MB per 128 tokens and cannot trigger the
                # MPSTemporaryNDArray hard limit on Apple Silicon.
                past_kv: DynamicCache = DynamicCache()
                out = model(
                    inputs_embeds=inputs_embeds,
                    position_ids=prefix_pos,
                    past_key_values=past_kv,
                    use_cache=True,
                )
                generated_ids: list[int] = []
                next_id = self._sample_next_token(out.logits[0, -1, :], generated_ids)
                past_kv = out.past_key_values

                # --- Step 6: Autoregressive generation loop (≤ 128 new tokens) ---
                # curr_pos starts immediately after the prefix so each token gets its
                # correct RoPE position independent of cache state queries.
                curr_pos = prefix_len
                _log("decode: starting manual generation loop (max_new_tokens=128)")
                for _step in range(128):
                    if next_id.item() == tokenizer.eos_token_id:
                        break
                    generated_ids.append(next_id.item())
                    loop_pos = torch.tensor(
                        [[curr_pos]], dtype=torch.long, device=device
                    )
                    out = model(
                        input_ids=next_id.unsqueeze(0),   # (1, 1)
                        position_ids=loop_pos,
                        past_key_values=past_kv,
                        use_cache=True,
                    )
                    next_id = self._sample_next_token(out.logits[0, -1, :], generated_ids)
                    past_kv = out.past_key_values
                    curr_pos += 1

                _log(f"decode: generation complete ({len(generated_ids)} tokens)")

        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
        _log(f"decode: tokenizer.decode complete ({len(decoded)} chars)")
        return decoded

    async def receive(self, framed_latent: "torch.Tensor") -> str:
        """Decode a framed latent sequence using the trained Base receiver.

        This is the Interlat-Lite replacement for decode().  It uses the Base
        model (LoRA fine-tuned to interpret latent injections) and does NOT
        apply NormMatch or Residual Anchoring — the trained model handles the
        manifold gap directly.

        Args:
            framed_latent: Tensor of shape (K+2, hidden_dim) — adapter output
                           already framed with <bop>/<eop> boundary embeddings.

        Returns:
            Decoded natural-language string.
        """
        import torch
        from transformers import DynamicCache

        model = self._mgr.get_base_model()
        tokenizer = self._mgr.get_base_tokenizer()
        device = self._mgr.device

        _log(f"receive: framed_latent {tuple(framed_latent.shape)}, device={device}")

        async with self._device_lock:
            with torch.no_grad():
                # Add batch dim: (1, K+2, D)
                embeds = framed_latent.to(device)
                if embeds.dim() == 2:
                    embeds = embeds.unsqueeze(0)

                # Prefix pass — inject framed latent directly, no preprocessing
                past_kv: DynamicCache = DynamicCache()
                prefix_len = embeds.shape[1]
                prefix_pos = torch.arange(
                    prefix_len, dtype=torch.long, device=device
                ).unsqueeze(0)

                out = model(
                    inputs_embeds=embeds,
                    position_ids=prefix_pos,
                    past_key_values=past_kv,
                    use_cache=True,
                )
                generated_ids: list[int] = []
                eos_id = tokenizer.eos_token_id
                min_new_tokens = 8  # suppress EOS until at least this many tokens

                def _suppress_eos(logits: "torch.Tensor", n_generated: int) -> "torch.Tensor":
                    if n_generated < min_new_tokens and eos_id is not None:
                        logits = logits.clone()
                        logits[eos_id] = float("-inf")
                    return logits

                first_logits = _suppress_eos(out.logits[0, -1, :], 0)
                next_id = self._sample_next_token(
                    first_logits, generated_ids,
                    temperature=self._receive_temperature,
                    top_p=self._receive_top_p,
                    top_k=self._receive_top_k,
                    repetition_penalty=self._receive_repetition_penalty,
                )
                past_kv = out.past_key_values

                # Autoregressive generation loop (≤ 512 new tokens)
                curr_pos = prefix_len
                for _step in range(512):
                    if next_id.item() == eos_id:
                        break
                    generated_ids.append(next_id.item())
                    loop_pos = torch.tensor([[curr_pos]], dtype=torch.long, device=device)
                    out = model(
                        input_ids=next_id.unsqueeze(0),
                        position_ids=loop_pos,
                        past_key_values=past_kv,
                        use_cache=True,
                    )
                    step_logits = _suppress_eos(out.logits[0, -1, :], len(generated_ids))
                    next_id = self._sample_next_token(
                        step_logits, generated_ids,
                        temperature=self._receive_temperature,
                        top_p=self._receive_top_p,
                        repetition_penalty=self._receive_repetition_penalty,
                    )
                    past_kv = out.past_key_values
                    curr_pos += 1

                _log(f"receive: generated {len(generated_ids)} tokens")

        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
        _log(f"receive: decode complete ({len(decoded)} chars)")
        return decoded
