"""LatentStrategy — V2 implementation using local HuggingFace transformers.

Librarians call encode() and reason() — both return torch.Tensor hidden states
from the model's last layer. No text is generated at these call sites.

ONLY the Translator calls decode() to produce natural-language output by
projecting hidden states through the LM head and running autoregressive generation.

See ARCHITECTURE.md §4 for the architectural constraints.
"""
from __future__ import annotations

import os
import sys

from libucks.thinking.base import ThinkingStrategy


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
        receive_temperature: float = 0.7,
        receive_top_p: float = 0.85,
        receive_top_k: int = 50,
        receive_repetition_penalty: float = 1.3,
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

        # Repetition penalty: count-proportional HuggingFace convention.
        # set() only penalises once per unique token, so a token appearing
        # 400 times is suppressed identically to one appearing once — causing
        # "the the the..." loops.  Applying penalty^count makes each additional
        # occurrence exponentially less likely.
        if _rep_penalty != 1.0 and generated_ids:
            from collections import Counter
            for token_id, count in Counter(generated_ids).items():
                effective = _rep_penalty ** min(count, 20)  # cap at 20 to avoid inf
                if logits[token_id] > 0:
                    logits[token_id] /= effective
                else:
                    logits[token_id] *= effective

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
                if inputs["input_ids"].shape[1] == 0:
                    raise ValueError(
                        f"encode() received empty tokenization (text={text!r:.80}). "
                        "Pass non-empty text."
                    )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                output = model(**inputs, output_hidden_states=True, use_cache=False)
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
                if seq_len == 0:
                    raise ValueError(
                        f"reason() received empty tokenization "
                        f"(context={context!r:.40}, query={query!r:.40}). "
                        "Pass non-empty context or query."
                    )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                _log(f"reason: forward pass (seq_len={seq_len})")
                output = model(**inputs, output_hidden_states=True, use_cache=False)
                hidden = output.hidden_states[-1].squeeze(0).contiguous()
                del output  # release full 36-layer hidden_states tuple
                _log(f"reason: forward pass complete, hidden={tuple(hidden.shape)}")

            if self._compressor is not None:
                _log(f"reason: compressing ({hidden.shape[0]} → {self._compressor.compression_steps} steps)")
                with torch.inference_mode():
                    hidden = self._compressor(hidden).contiguous()
                _log(f"reason: compressed to {tuple(hidden.shape)}")

        return hidden

    async def decode(self, result: "torch.Tensor", query: str = "", verbatim: str = "") -> str:  # noqa: ARG002
        """Convert an adapter soft-prompt into natural language via the trained Base receiver.

        This is the Interlat-Lite decoder. It:
          1. Frames the soft-prompt with <|im_start|>/<|im_end|> boundary embeddings
             (token recycling — native Qwen boundary markers as <bop>/<eop>).
          2. Runs nucleus sampling (do_sample=True) via model.generate() using the
             LoRA-trained Qwen2.5-0.5B-Base receiver.

        Beam search (num_beams=4) is incompatible with 4-bit MPS: batch expansion
        to beam_count collapses all beams to EOS after 1 token on mps_bitsandbytes
        Linear4bit layers. Nucleus sampling avoids batch expansion and achieves the
        same anti-greedy-argmax goal.

        The trained model handles the Instruct→Base manifold gap; no NormMatch or
        Residual Anchoring is applied here.

        This is the ONLY authorised call site for generative inference in the system.

        Args:
            result: torch.Tensor of shape (K, hidden_dim) — the adapter soft-prompt.

        Returns:
            Decoded natural-language string.
        """
        import os
        import torch

        model = self._mgr.get_base_model()
        tokenizer = self._mgr.get_base_tokenizer()
        device = self._mgr.device

        _log(f"decode: received tensor {tuple(result.shape)}, device={device}")

        async with self._device_lock:
            with torch.no_grad():
                # --- Frame with <bop>/<eop> boundary embeddings ---
                # Token recycling: native Qwen chat-boundary tokens as frame markers.
                bop_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
                eop_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
                embedding_layer = model.model.embed_tokens
                bop_embed = embedding_layer(
                    torch.tensor([bop_id], device=device)
                ).squeeze(0).detach()
                eop_embed = embedding_layer(
                    torch.tensor([eop_id], device=device)
                ).squeeze(0).detach()

                soft_prompt = result.to(device)
                if soft_prompt.dim() == 3:
                    soft_prompt = soft_prompt.squeeze(0)   # (K, d)

                # Capture raw adapter output (pre-W_a) for diagnostics
                _adapter_raw = soft_prompt.clone().detach()

                # Mirror the training transform exactly (see _cli.py:_train_lora_receiver):
                #   1. Subtract population soft_mean (center) — removes DC component
                #   2. Project via W_a — aligns hidden-state space → embedding space
                #   3. Norm-rescale to embed_norm — matches scale seen during LoRA training
                # W_a and soft_mean are loaded from w_a.pt by _load_lora_weights().
                model_dtype = embedding_layer.weight.dtype
                embed_norm = embedding_layer.weight.data.norm(dim=-1).median()
                sp_f32 = soft_prompt.float()
                # Centering (subtracting _soft_mean) was removed: with the new
                # diverse adapter (post-residual-fix), per-slot mean subtraction
                # collapsed inter-slot cosine from ~0.50 back to ~0.97 by
                # stripping the slot identities themselves. W_a remains since
                # it's identity-like for tied vocab and a no-op otherwise.
                if hasattr(self, "_W_a") and self._W_a is not None:
                    sp_f32 = sp_f32 @ self._W_a.to(device)
                sp_norms = sp_f32.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                soft_prompt = (sp_f32 / sp_norms * embed_norm).to(model_dtype)

                import os
                if os.environ.get("LIBUCKS_BLINDFOLD") == "1":
                    _log("BLINDFOLD: replacing soft_prompt with norm-matched random noise")
                    noise = torch.randn_like(soft_prompt)
                    noise = noise / noise.norm(dim=-1, keepdim=True).clamp(min=1e-8) * embed_norm
                    soft_prompt = noise.to(model_dtype)

                framed = torch.cat(
                    [bop_embed.view(1, -1), soft_prompt, eop_embed.view(1, -1)], dim=0
                )  # (K+2, d)

                # Hybrid retrieval: prepend verbatim source code embeddings
                # before <bop>. Truncated at 2048 tokens — under 7% of Qwen's
                # 32K context. Source code grounds the soft prompt's
                # cross-bucket reasoning at the identifier level.
                if verbatim:
                    v_enc = tokenizer(
                        verbatim, return_tensors="pt", truncation=True,
                        max_length=2048, add_special_tokens=False,
                    )
                    v_ids = v_enc["input_ids"].squeeze(0).to(device)
                    v_embeds = embedding_layer(v_ids).to(model_dtype)
                    framed = torch.cat([v_embeds, framed], dim=0)
                    _log(f"decode: hybrid verbatim prepended ({v_ids.shape[0]} tokens)")

                # Append query tokens — matches training frame exactly.
                # Truncated to 32 tokens to bound sequence length.
                if query:
                    q_enc = tokenizer(
                        query, return_tensors="pt", truncation=True, max_length=32,
                        add_special_tokens=False,
                    )
                    q_ids = q_enc["input_ids"].squeeze(0).to(device)
                    q_embeds = embedding_layer(q_ids).to(model_dtype)
                    framed = torch.cat([framed, q_embeds], dim=0)  # (K+2+Q, d)

                # Append assistant turn-start for the Instruct model.
                # Without this cue, the Instruct model treats the framed input as a
                # corrupted continuation and produces incoherent output. The token
                # sequence "<|im_start|>assistant\n" signals it should generate a
                # structured answer, activating its instruction-following prior.
                asst_enc = tokenizer(
                    "<|im_start|>assistant\n",
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                asst_ids = asst_enc["input_ids"].squeeze(0).to(device)
                asst_embeds = embedding_layer(asst_ids).to(model_dtype)
                framed = torch.cat([framed, asst_embeds], dim=0)

                # Add batch dim: (1, K+2+Q+A, D)
                embeds = framed.unsqueeze(0)

                # Nucleus sampling via HuggingFace generate(). attention_mask: explicit
                # all-ones required — Qwen sets pad_token_id == eos_token_id, which
                # prevents HF 5.x from inferring the mask and raises a generation error.
                # min_new_tokens=8 suppresses EOS for the first 8 tokens to prevent
                # immediate collapse (same role as the old _suppress_eos while-loop guard).
                attention_mask = torch.ones(
                    1, embeds.shape[1], dtype=torch.long, device=device
                )

                # ─── ATTENTION DEBUG (LIBUCKS_ATTENTION_DEBUG=1) ──────────────
                # Diagnose where the model is looking and whether the soft-prompt
                # vectors are mathematically sensible (on-manifold) or random.
                if os.environ.get("LIBUCKS_ATTENTION_DEBUG") == "1":
                    K = soft_prompt.shape[0]
                    Q = q_embeds.shape[0] if query else 0
                    A = asst_embeds.shape[0]
                    S = embeds.shape[1]
                    bop_idx = 0
                    slot_lo, slot_hi = 1, 1 + K
                    eop_idx = 1 + K
                    q_lo, q_hi = 2 + K, 2 + K + Q
                    cue_lo, cue_hi = 2 + K + Q, S
                    _log(f"DEBUG layout: bop=[0]  slots=[{slot_lo}..{slot_hi-1}] (K={K})  "
                         f"eop=[{eop_idx}]  query=[{q_lo}..{q_hi-1}] (Q={Q})  "
                         f"cue=[{cue_lo}..{cue_hi-1}] (A={A})  S={S}")

                    # (1) Soft-prompt geometry
                    sp_f = soft_prompt.float()                               # (K, D)
                    sp_norms = sp_f.norm(dim=-1)                             # (K,)
                    sp_unit = sp_f / sp_norms.unsqueeze(-1).clamp(min=1e-8)
                    vocab_emb = embedding_layer.weight.float()               # (V, D)
                    vocab_unit = vocab_emb / vocab_emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    cos_to_vocab = sp_unit @ vocab_unit.T                    # (K, V)
                    max_cos, max_idx = cos_to_vocab.max(dim=-1)              # (K,) (K,)
                    pair_cos = (sp_unit @ sp_unit.T)                         # (K, K)
                    pair_off = pair_cos - torch.eye(K, device=pair_cos.device)
                    nearest_tokens = [tokenizer.decode([i.item()]) for i in max_idx]

                    _log(f"DEBUG soft-prompt norms: min={sp_norms.min():.3f}  "
                         f"mean={sp_norms.mean():.3f}  max={sp_norms.max():.3f}  "
                         f"baseline embed_norm={embed_norm.item():.3f}")

                    # Pre-W_a (raw adapter output) inter-slot cosine — localizes the
                    # collapse to either the adapter or the W_a transform.
                    _raw = _adapter_raw.float()
                    _raw_unit = _raw / _raw.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    _raw_pair = (_raw_unit @ _raw_unit.T) - torch.eye(K, device=_raw.device)
                    _log(f"DEBUG PRE-W_a inter-slot cos: mean={_raw_pair.mean():.4f}  "
                         f"max={_raw_pair.max():.4f}  "
                         f"raw_norm_mean={_raw.norm(dim=-1).mean():.3f}")
                    _log(f"DEBUG cos-to-nearest-vocab: min={max_cos.min():.4f}  "
                         f"mean={max_cos.mean():.4f}  max={max_cos.max():.4f}  "
                         f"(real-token cos≈1.0; off-manifold ≪ 0.1)")
                    _log(f"DEBUG inter-slot cos (off-diag): mean={pair_off.mean():.4f}  "
                         f"max={pair_off.max():.4f}  (low = diverse slots; high = collapsed)")
                    _log(f"DEBUG slot[0..7] nearest vocab: {[t for t in nearest_tokens[:8]]}")

                    # (2) Attention mass per region from the position that will
                    # predict the first generated token (last assistant-cue token)
                    try:
                        out_attn = model(inputs_embeds=embeds,
                                         attention_mask=attention_mask,
                                         use_cache=False, output_attentions=True)
                        attns = out_attn.attentions  # tuple of L × (1, H, S, S)
                        L = len(attns)
                        H = attns[0].shape[1]
                        avg_attn = torch.stack([a[0].float().mean(dim=0) for a in attns]).mean(dim=0)
                        # ^ (S, S) — averaged over heads and layers
                        cue_last = S - 1
                        row = avg_attn[cue_last]                              # (S,)
                        m_bop = row[bop_idx].item()
                        m_slot = row[slot_lo:slot_hi].sum().item()
                        m_eop = row[eop_idx].item()
                        m_query = row[q_lo:q_hi].sum().item() if Q > 0 else 0.0
                        m_cue = row[cue_lo:cue_hi].sum().item()
                        tot = m_bop + m_slot + m_eop + m_query + m_cue
                        _log(f"DEBUG attention from cue[{cue_last}] (avg over L={L}, H={H}):")
                        _log(f"  bop   {m_bop:.4f} ({100*m_bop/tot:5.1f}%)")
                        _log(f"  slots {m_slot:.4f} ({100*m_slot/tot:5.1f}%)  "
                             f"per-slot mean={m_slot/K:.4f}")
                        _log(f"  eop   {m_eop:.4f} ({100*m_eop/tot:5.1f}%)")
                        _log(f"  query {m_query:.4f} ({100*m_query/tot:5.1f}%)  "
                             f"per-token mean={(m_query/Q if Q>0 else 0):.4f}")
                        _log(f"  cue   {m_cue:.4f} ({100*m_cue/tot:5.1f}%)")
                        del out_attn, attns, avg_attn
                    except Exception as _e:
                        _log(f"DEBUG attention extraction FAILED: {type(_e).__name__}: {_e}")

                # repetition_penalty crashes on MPS (RepetitionPenaltyLogitsProcessor
                # triggers MPSTemporaryNDArray > 4 GB). Explicitly set to 1.0 to
                # override the Instruct model's generation_config.json default (1.1).
                # no_repeat_ngram_size=3 is a CPU-side logit processor that prevents
                # 3-gram repeats — same anti-loop protection without MPS allocation.
                _gen_temperature = float(os.environ.get("LIBUCKS_GEN_TEMPERATURE", self._receive_temperature))
                _gen_top_p = float(os.environ.get("LIBUCKS_GEN_TOP_P", self._receive_top_p))
                _gen_top_k = int(os.environ.get("LIBUCKS_GEN_TOP_K", self._receive_top_k))
                _gen_ngram = int(os.environ.get("LIBUCKS_GEN_NGRAM", "3"))
                _gen_min_new = int(os.environ.get("LIBUCKS_GEN_MIN_NEW", "8"))
                if any(k in os.environ for k in (
                    "LIBUCKS_GEN_TEMPERATURE", "LIBUCKS_GEN_TOP_P", "LIBUCKS_GEN_TOP_K",
                    "LIBUCKS_GEN_NGRAM", "LIBUCKS_GEN_MIN_NEW",
                )):
                    _log(
                        f"decode: gen_overrides T={_gen_temperature} top_p={_gen_top_p} "
                        f"top_k={_gen_top_k} ngram={_gen_ngram} min_new={_gen_min_new}"
                    )
                output_ids = model.generate(
                    inputs_embeds=embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=256,
                    min_new_tokens=_gen_min_new,
                    do_sample=True,
                    temperature=_gen_temperature,
                    top_p=_gen_top_p,
                    top_k=_gen_top_k,
                    repetition_penalty=1.0,
                    no_repeat_ngram_size=_gen_ngram,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                # generate() with inputs_embeds returns only the newly generated
                # token IDs — the embedded prefix is not echoed back as tokens.
                _log(f"decode: generated {output_ids.shape[1]} tokens")

        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        _log(f"decode: decode complete ({len(decoded)} chars)")
        return decoded
