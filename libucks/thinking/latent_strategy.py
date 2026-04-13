"""LatentStrategy — V2 implementation using local HuggingFace transformers.

Librarians call encode() and reason() — both return torch.Tensor hidden states
from the model's last layer. No text is generated at these call sites.

ONLY the Translator calls decode() to produce natural-language output by
projecting hidden states through the LM head and running autoregressive generation.

See ARCHITECTURE.md §4 for the architectural constraints.
"""
from __future__ import annotations

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

    async def decode(self, result: "torch.Tensor", query: str = "") -> str:
        """Convert an adapter soft-prompt into natural language via the trained Base receiver.

        This is the Interlat-Lite decoder. It:
          1. Frames the soft-prompt with <|im_start|>/<|im_end|> boundary embeddings
             (token recycling — native Qwen boundary markers as <bop>/<eop>).
          2. Runs autoregressive generation using the LoRA-trained Qwen2.5-3B-Base
             receiver, which has learned to interpret latent injections directly.

        The trained model handles the Instruct→Base manifold gap; no NormMatch or
        Residual Anchoring is applied here.

        This is the ONLY authorised call site for generative inference in the system.

        Args:
            result: torch.Tensor of shape (K, hidden_dim) — the adapter soft-prompt.

        Returns:
            Decoded natural-language string.
        """
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

                # Scale-normalize to the Base model's native embedding scale.
                # The adapter output lives at Instruct hidden-state scale (~10-50/dim);
                # the Base model's embed_tokens operates at ~2-3/dim. Training in
                # _cli.py:_train_lora_receiver() rescaled every soft-prompt to
                # embed_norm before the model ever saw it. Without mirroring that
                # here, Q×K products in attention overflow → catastrophic degeneration.
                # Cast to model dtype afterward so torch.cat with bop/eop (float16)
                # does not produce a dtype mismatch.
                model_dtype = embedding_layer.weight.dtype
                embed_norm = embedding_layer.weight.data.norm(dim=-1).median()
                sp_norms = soft_prompt.float().norm(dim=-1, keepdim=True).clamp(min=1e-8)
                soft_prompt = (soft_prompt.float() / sp_norms * embed_norm).to(model_dtype)

                framed = torch.cat(
                    [bop_embed.view(1, -1), soft_prompt, eop_embed.view(1, -1)], dim=0
                )  # (K+2, d)

                # Append query embeddings so the receiver is conditioned on the
                # user's question at inference time, matching the training frame
                # [BOP][latent(K)][EOP][query_toks(Q)].
                if query:
                    q_enc = tokenizer(
                        query, return_tensors="pt", truncation=True, max_length=32
                    )
                    q_ids = q_enc["input_ids"].squeeze(0).to(device)       # (Q,)
                    q_embeds = embedding_layer(q_ids).to(model_dtype)       # (Q, D)
                    framed = torch.cat([framed, q_embeds], dim=0)           # (K+2+Q, d)

                # Add batch dim: (1, K+2+Q, D)
                embeds = framed.unsqueeze(0)

                # Prefix pass — inject framed latent directly, no preprocessing
                past_kv = None
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

                _log(f"decode: generated {len(generated_ids)} tokens")

        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
        _log(f"decode: decode complete ({len(decoded)} chars)")
        return decoded
