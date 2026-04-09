"""Phase 8: LatentStrategy — full tensor implementation tests.

All model calls are mocked so no GPU or real model download is required.
The mock returns deterministic torch.Tensor shapes matching Qwen2.5-3B-Instruct:
  hidden_dim = 2048, num_layers = 28.
"""
from __future__ import annotations

import pytest
import torch
from unittest.mock import MagicMock, call

from libucks.thinking.latent_strategy import LatentStrategy


# ---------------------------------------------------------------------------
# Constants matching Qwen2.5-3B-Instruct architecture
# ---------------------------------------------------------------------------
HIDDEN_DIM = 2048
NUM_LAYERS = 28
VOCAB_SIZE = 32000
SEQ_LEN = 10
NEW_TOKENS = 5
# The mock tokenizer always returns SEQ_LEN tokens regardless of input text,
# so the anchor prompt also tokenises to SEQ_LEN tokens in tests.
ANCHOR_LEN = SEQ_LEN


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_model_output(seq_len: int = SEQ_LEN) -> MagicMock:
    """Mock CausalLMOutputWithCrossAttentions with realistic hidden_states tuple."""
    hidden_states = tuple(
        torch.randn(1, seq_len, HIDDEN_DIM) for _ in range(NUM_LAYERS + 1)
    )
    output = MagicMock()
    output.hidden_states = hidden_states
    return output


def _make_mock_mgr(seq_len: int = SEQ_LEN) -> MagicMock:
    """Return a fully-configured mock ModelManager.

    Supports both encode/reason calls (output_hidden_states=True → hidden states)
    and decode calls (no output_hidden_states → logits output).  embed_tokens is
    wired so the new NormMatch + inputs_embeds decode path works.
    """
    mgr = MagicMock()
    mgr.device = "cpu"

    # --- tokenizer ---
    mock_tokenizer = MagicMock()
    input_ids = torch.randint(0, VOCAB_SIZE, (1, seq_len))
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)
    mock_tokenizer.return_value = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    mock_tokenizer.eos_token_id = 2
    mock_tokenizer.decode.return_value = "The authentication module validates JWT tokens."
    mgr.get_tokenizer.return_value = mock_tokenizer

    # --- embed_tokens (required by NormMatch in decode()) ---
    embed_tokens = MagicMock()
    embed_tokens.weight = torch.randn(VOCAB_SIZE, HIDDEN_DIM)
    # Returns (1, ANCHOR_LEN, HIDDEN_DIM) — same length as tokenizer mock output
    embed_tokens.return_value = torch.randn(1, ANCHOR_LEN, HIDDEN_DIM)

    # --- model ---
    mock_model = MagicMock()
    mock_model.model = MagicMock()
    mock_model.model.embed_tokens = embed_tokens

    # lm_head kept as a bare MagicMock so assert_not_called() works in decode tests.
    mock_model.lm_head = MagicMock()
    mock_model.lm_head.return_value = torch.randn(1, seq_len, VOCAB_SIZE)

    # generate kept for backward-compat assertions in encode/reason tests.
    mock_model.generate.return_value = torch.randint(0, VOCAB_SIZE, (1, seq_len + NEW_TOKENS))

    # Side-effect routes calls by purpose:
    #   encode/reason → output_hidden_states=True → return hidden states mock
    #   decode        → no output_hidden_states   → return logits mock (non-EOS)
    def _model_side_effect(*args, **kwargs):
        if kwargs.get("output_hidden_states"):
            return _make_model_output(seq_len)
        # decode path: return logits where argmax = 100 (never eos_token_id=2)
        out = MagicMock()
        logits = torch.full((1, 1, VOCAB_SIZE), -1.0)
        logits[0, 0, 100] = 10.0
        out.logits = logits
        out.past_key_values = MagicMock()
        return out

    mock_model.side_effect = _model_side_effect

    mgr.get_model.return_value = mock_model
    return mgr


@pytest.fixture
def mgr() -> MagicMock:
    return _make_mock_mgr()


@pytest.fixture
def strategy(mgr: MagicMock) -> LatentStrategy:
    return LatentStrategy(mgr)


# ---------------------------------------------------------------------------
# encode()
# ---------------------------------------------------------------------------

class TestLatentStrategyEncode:
    async def test_encode_returns_tensor(self, strategy):
        result = await strategy.encode("hello world")
        assert isinstance(result, torch.Tensor)

    async def test_encode_shape_is_seq_len_by_hidden_dim(self, strategy):
        result = await strategy.encode("hello world")
        assert result.dim() == 2
        assert result.shape == (SEQ_LEN, HIDDEN_DIM)

    async def test_encode_does_not_call_generate(self, strategy, mgr):
        await strategy.encode("hello world")
        mgr.get_model().generate.assert_not_called()

    async def test_encode_calls_model_with_output_hidden_states(self, strategy, mgr):
        await strategy.encode("hello world")
        call_kwargs = mgr.get_model().call_args[1]
        assert call_kwargs.get("output_hidden_states") is True

    async def test_encode_tokenizes_the_input_text(self, strategy, mgr):
        await strategy.encode("unique_encode_text")
        tokenizer_call_args = mgr.get_tokenizer().call_args
        assert tokenizer_call_args[0][0] == "unique_encode_text"

    async def test_encode_requests_pt_tensors_from_tokenizer(self, strategy, mgr):
        await strategy.encode("some text")
        tokenizer_call_kwargs = mgr.get_tokenizer().call_args[1]
        assert tokenizer_call_kwargs.get("return_tensors") == "pt"

    async def test_encode_empty_string_does_not_raise(self, strategy):
        result = await strategy.encode("")
        assert isinstance(result, torch.Tensor)

    async def test_encode_uses_last_hidden_layer(self, strategy):
        """encode() must return a tensor with the last-layer shape (seq_len, hidden_dim)."""
        result = await strategy.encode("test")
        assert result.shape == (SEQ_LEN, HIDDEN_DIM)


# ---------------------------------------------------------------------------
# reason()
# ---------------------------------------------------------------------------

class TestLatentStrategyReason:
    async def test_reason_returns_tensor(self, strategy):
        result = await strategy.reason("what is auth?", "JWT context")
        assert isinstance(result, torch.Tensor)

    async def test_reason_shape_is_seq_len_by_hidden_dim(self, strategy):
        result = await strategy.reason("query", "context")
        assert result.dim() == 2
        assert result.shape == (SEQ_LEN, HIDDEN_DIM)

    async def test_reason_does_not_call_generate(self, strategy, mgr):
        await strategy.reason("query", "context")
        mgr.get_model().generate.assert_not_called()

    async def test_reason_prompt_includes_query(self, strategy, mgr):
        await strategy.reason("unique_query_marker", "some context")
        tokenizer_call_args = mgr.get_tokenizer().call_args
        prompt = tokenizer_call_args[0][0]
        assert "unique_query_marker" in prompt

    async def test_reason_prompt_includes_context(self, strategy, mgr):
        await strategy.reason("some query", "unique_context_marker")
        tokenizer_call_args = mgr.get_tokenizer().call_args
        prompt = tokenizer_call_args[0][0]
        assert "unique_context_marker" in prompt

    async def test_reason_empty_context_does_not_raise(self, strategy):
        result = await strategy.reason("query", "")
        assert isinstance(result, torch.Tensor)

    async def test_reason_calls_model_with_output_hidden_states(self, strategy, mgr):
        await strategy.reason("q", "ctx")
        call_kwargs = mgr.get_model().call_args[1]
        assert call_kwargs.get("output_hidden_states") is True

    async def test_reason_uses_last_hidden_layer(self, strategy):
        """reason() must return a tensor with the last-layer shape (seq_len, hidden_dim)."""
        result = await strategy.reason("q", "ctx")
        assert result.shape == (SEQ_LEN, HIDDEN_DIM)


# ---------------------------------------------------------------------------
# decode() — interface contracts that survive the NormMatch rewrite
# ---------------------------------------------------------------------------

class TestLatentStrategyDecode:
    async def test_decode_returns_str(self, strategy):
        tensor = torch.randn(SEQ_LEN, HIDDEN_DIM)
        result = await strategy.decode(tensor)
        assert isinstance(result, str)

    async def test_decode_calls_tokenizer_decode(self, strategy, mgr):
        tensor = torch.randn(SEQ_LEN, HIDDEN_DIM)
        await strategy.decode(tensor)
        mgr.get_tokenizer().decode.assert_called_once()

    async def test_decode_result_matches_tokenizer_output(self, strategy, mgr):
        mgr.get_tokenizer().decode.return_value = "Expected answer string."
        tensor = torch.randn(SEQ_LEN, HIDDEN_DIM)
        result = await strategy.decode(tensor)
        assert result == "Expected answer string."

    async def test_decode_passes_skip_special_tokens(self, strategy, mgr):
        tensor = torch.randn(SEQ_LEN, HIDDEN_DIM)
        await strategy.decode(tensor)
        decode_call_kwargs = mgr.get_tokenizer().decode.call_args[1]
        assert decode_call_kwargs.get("skip_special_tokens") is True

    async def test_decode_accepts_batched_tensor(self, strategy):
        """decode() should handle both (seq_len, d) and (1, seq_len, d) input."""
        tensor_2d = torch.randn(SEQ_LEN, HIDDEN_DIM)
        tensor_3d = torch.randn(1, SEQ_LEN, HIDDEN_DIM)
        result_2d = await strategy.decode(tensor_2d)
        result_3d = await strategy.decode(tensor_3d)
        assert isinstance(result_2d, str)
        assert isinstance(result_3d, str)


# ---------------------------------------------------------------------------
# decode() — NormMatch + inputs_embeds injection (new contract)
# ---------------------------------------------------------------------------

class TestLatentStrategyDecodeNormMatch:
    """Tests for the NormMatch + inputs_embeds injection path.

    Written BEFORE the implementation (TDD).  All tests in this class FAIL
    on the old code which uses lm_head + argmax.
    """

    async def test_anchor_prompt_constant_is_module_level(self):
        """_ANCHOR_PROMPT must be defined at module level in latent_strategy
        and must use the Qwen2.5 chat-template assistant marker so the instruct
        model activates its instruction-following mode."""
        import libucks.thinking.latent_strategy as mod
        assert hasattr(mod, "_ANCHOR_PROMPT")
        assert isinstance(mod._ANCHOR_PROMPT, str)
        assert len(mod._ANCHOR_PROMPT) > 0
        assert "<|im_start|>assistant" in mod._ANCHOR_PROMPT, (
            "_ANCHOR_PROMPT must contain '<|im_start|>assistant' to trigger "
            "Qwen2.5-instruct's instruction-following mode"
        )

    async def test_decode_does_not_call_lm_head(self, strategy, mgr):
        """Core constraint: lm_head must never be called in the new path."""
        tensor = torch.randn(SEQ_LEN, HIDDEN_DIM)
        await strategy.decode(tensor)
        mgr.get_model().lm_head.assert_not_called()

    async def test_decode_first_model_call_uses_inputs_embeds(self, strategy, mgr):
        """First model() call must use inputs_embeds keyword arg."""
        tensor = torch.randn(SEQ_LEN, HIDDEN_DIM)
        await strategy.decode(tensor)
        first_call = mgr.get_model().call_args_list[0]
        assert "inputs_embeds" in first_call.kwargs, (
            "First model() call must pass inputs_embeds= keyword argument"
        )

    async def test_decode_first_model_call_has_no_input_ids(self, strategy, mgr):
        """First model() call must NOT pass input_ids (that is the old broken path)."""
        tensor = torch.randn(SEQ_LEN, HIDDEN_DIM)
        await strategy.decode(tensor)
        first_call = mgr.get_model().call_args_list[0]
        assert "input_ids" not in first_call.kwargs, (
            "First model() call must not pass input_ids alongside inputs_embeds"
        )

    async def test_decode_subsequent_loop_calls_use_input_ids(self, strategy, mgr):
        """After the prefix pass, every loop iteration must use input_ids (single token)."""
        tensor = torch.randn(SEQ_LEN, HIDDEN_DIM)
        await strategy.decode(tensor)
        calls = mgr.get_model().call_args_list
        assert len(calls) >= 2, "Expected at least one autoregressive step"
        for loop_call in calls[1:]:
            assert "input_ids" in loop_call.kwargs, (
                f"Loop call must use input_ids, got kwargs: {list(loop_call.kwargs)}"
            )
            assert "inputs_embeds" not in loop_call.kwargs, (
                "Loop call must not use inputs_embeds"
            )

    async def test_decode_normatch_rescales_to_embed_scale(self, strategy, mgr):
        """NormMatch: the soft-prompt portion of inputs_embeds must have per-row
        norms equal to embed_tokens.weight.norm(dim=-1).mean() within 1%."""
        embed_scale = (
            mgr.get_model().model.embed_tokens.weight
            .norm(dim=-1).mean().item()
        )
        tensor = torch.randn(SEQ_LEN, HIDDEN_DIM)
        await strategy.decode(tensor)
        first_call = mgr.get_model().call_args_list[0]
        inputs_embeds = first_call.kwargs["inputs_embeds"]  # (1, K+L, d)
        soft_prompt = inputs_embeds[0, :SEQ_LEN, :]         # (K, d)
        mean_norm = soft_prompt.norm(dim=-1).mean().item()
        assert abs(mean_norm - embed_scale) / embed_scale < 0.01, (
            f"NormMatch failed: mean soft-prompt norm={mean_norm:.4f}, "
            f"embed_scale={embed_scale:.4f}"
        )

    async def test_decode_anchor_tokenized_without_special_tokens(self, strategy, mgr):
        """Anchor tokenizer call must pass add_special_tokens=False.

        Without it, Qwen's tokenizer prepends BOS before <|im_start|>, which
        shifts all anchor position IDs by +1 and inserts a confusing restart
        signal at position K in the DynamicCache.
        """
        tensor = torch.randn(SEQ_LEN, HIDDEN_DIM)
        await strategy.decode(tensor)
        tokenizer_calls = mgr.get_tokenizer().call_args_list
        anchor_calls = [
            c for c in tokenizer_calls
            if c.kwargs.get("add_special_tokens") is False
        ]
        assert len(anchor_calls) >= 1, (
            "Anchor tokenizer call must pass add_special_tokens=False"
        )

    async def test_decode_anchor_embeddings_appended_after_soft_prompt(
        self, strategy, mgr
    ):
        """inputs_embeds must be [soft_prompt | anchor_embeds] with shape
        (1, SEQ_LEN + ANCHOR_LEN, HIDDEN_DIM).  embed_tokens is called twice:
        once for the dummy baseline and once for the anchor prompt."""
        tensor = torch.randn(SEQ_LEN, HIDDEN_DIM)
        await strategy.decode(tensor)
        first_call = mgr.get_model().call_args_list[0]
        inputs_embeds = first_call.kwargs["inputs_embeds"]
        assert mgr.get_model().model.embed_tokens.call_count == 2, (
            "embed_tokens must be called twice: dummy baseline + anchor prompt"
        )
        assert inputs_embeds.shape == (1, SEQ_LEN + ANCHOR_LEN, HIDDEN_DIM), (
            f"Expected (1, {SEQ_LEN + ANCHOR_LEN}, {HIDDEN_DIM}), "
            f"got {tuple(inputs_embeds.shape)}"
        )

    async def test_decode_anchor_portion_matches_embed_tokens_output(
        self, strategy, mgr
    ):
        """The anchor rows of inputs_embeds must equal embed_tokens(anchor_ids)."""
        anchor_embeds = torch.randn(1, ANCHOR_LEN, HIDDEN_DIM)
        mgr.get_model().model.embed_tokens.return_value = anchor_embeds
        tensor = torch.randn(SEQ_LEN, HIDDEN_DIM)
        await strategy.decode(tensor)
        first_call = mgr.get_model().call_args_list[0]
        inputs_embeds = first_call.kwargs["inputs_embeds"]
        anchor_portion = inputs_embeds[0, SEQ_LEN:, :]  # (ANCHOR_LEN, d)
        assert torch.allclose(anchor_portion, anchor_embeds.squeeze(0)), (
            "Anchor portion of inputs_embeds does not match embed_tokens output"
        )

    async def test_decode_uses_dynamic_cache_on_first_call(self, strategy, mgr):
        """First model() call must receive a fresh DynamicCache as past_key_values."""
        from transformers import DynamicCache
        tensor = torch.randn(SEQ_LEN, HIDDEN_DIM)
        await strategy.decode(tensor)
        first_call = mgr.get_model().call_args_list[0]
        pkv = first_call.kwargs.get("past_key_values")
        assert pkv is not None, "past_key_values must be passed on first call"
        assert isinstance(pkv, DynamicCache), (
            f"Expected DynamicCache on first call, got {type(pkv)}"
        )

    async def test_decode_passes_use_cache_true_on_all_calls(self, strategy, mgr):
        """use_cache=True must be passed on every model() call."""
        tensor = torch.randn(SEQ_LEN, HIDDEN_DIM)
        await strategy.decode(tensor)
        for i, c in enumerate(mgr.get_model().call_args_list):
            assert c.kwargs.get("use_cache") is True, (
                f"Call {i} missing use_cache=True"
            )

    async def test_decode_respects_max_new_tokens_128_cap(self, strategy, mgr):
        """Generation must stop after at most 128 new tokens (never exceeds cap)."""
        tensor = torch.randn(SEQ_LEN, HIDDEN_DIM)
        await strategy.decode(tensor)
        # call 0 = prefix pass; calls 1..N = loop iterations
        loop_calls = mgr.get_model().call_args_list[1:]
        assert len(loop_calls) <= 128, (
            f"Expected ≤128 loop iterations, got {len(loop_calls)}"
        )

    async def test_decode_stops_at_eos_token(self, strategy, mgr):
        """Generation must stop immediately when EOS token is produced."""
        eos_id = mgr.get_tokenizer().eos_token_id  # 2
        prefix_len = SEQ_LEN + ANCHOR_LEN

        # Prefix pass: first generated token is 100 (non-EOS)
        first_logits = torch.full((1, prefix_len, VOCAB_SIZE), -1.0)
        first_logits[0, -1, 100] = 10.0
        first_out = MagicMock()
        first_out.logits = first_logits
        first_out.past_key_values = MagicMock()

        # First loop step: model produces EOS
        eos_logits = torch.full((1, 1, VOCAB_SIZE), -1.0)
        eos_logits[0, 0, eos_id] = 10.0
        eos_out = MagicMock()
        eos_out.logits = eos_logits
        eos_out.past_key_values = MagicMock()

        mgr.get_model().side_effect = [first_out, eos_out]

        tensor = torch.randn(SEQ_LEN, HIDDEN_DIM)
        result = await strategy.decode(tensor)

        # call 0 = prefix pass, call 1 = loop step that produced EOS → break
        assert mgr.get_model().call_count == 2
        assert isinstance(result, str)

    async def test_decode_returns_str_after_normatch(self, strategy):
        """End-to-end: decode() still returns a str with the new flow."""
        tensor = torch.randn(SEQ_LEN, HIDDEN_DIM)
        result = await strategy.decode(tensor)
        assert isinstance(result, str)

    async def test_decode_accepts_batched_tensor_normatch(self, strategy):
        """Both (K, d) and (1, K, d) inputs must work with the new flow."""
        tensor_2d = torch.randn(SEQ_LEN, HIDDEN_DIM)
        tensor_3d = torch.randn(1, SEQ_LEN, HIDDEN_DIM)
        r2 = await strategy.decode(tensor_2d)
        r3 = await strategy.decode(tensor_3d)
        assert isinstance(r2, str)
        assert isinstance(r3, str)


# ---------------------------------------------------------------------------
# decode() — Residual Anchoring (dummy baseline + gate + position IDs)
# ---------------------------------------------------------------------------

class TestLatentStrategyDecodeResidualAnchoring:
    """Tests for Text-Based Residual Injection (Vision Wormhole Eq. 2).

    Written BEFORE the implementation (TDD).  All tests that exercise new
    behaviour FAIL on the current code.
    """

    async def test_decode_embed_tokens_called_twice(self, strategy, mgr):
        """embed_tokens must be called twice: once for the dummy baseline,
        once for the anchor prompt."""
        tensor = torch.randn(SEQ_LEN, HIDDEN_DIM)
        await strategy.decode(tensor)
        assert mgr.get_model().model.embed_tokens.call_count == 2

    async def test_decode_dummy_baseline_uses_space_token(self, strategy, mgr):
        """Residual Anchoring dummy baseline must use the space token, not eos_token_id.

        eos_token_id in Qwen2.5 is <|im_end|> — embedding K copies of this as
        the baseline creates a 'K completed conversation turns' prefix that
        conflicts with the ChatML anchor and produces character-soup output.
        The space token is semantically neutral with no conversation-structure prior.
        """
        tensor = torch.randn(SEQ_LEN, HIDDEN_DIM)
        await strategy.decode(tensor)
        tok = mgr.get_tokenizer()
        # Indirect check: tokenizer(" ", add_special_tokens=False) must have been called
        # to resolve the space token ID for the dummy baseline.
        space_calls = [
            c for c in tok.call_args_list
            if c.args and c.args[0] == " "
        ]
        assert len(space_calls) >= 1, (
            "decode() must call tokenizer(' ', add_special_tokens=False) to resolve "
            "the space token ID for the dummy baseline."
        )

    async def test_decode_zero_gate_gives_dummy_embeddings(self, mgr):
        """At injection_gate=0 the soft-prompt must equal the (re-normalised)
        dummy baseline — the adapter signal is completely gated off."""
        embed_scale = mgr.get_model().model.embed_tokens.weight.norm(dim=-1).mean()

        # Build dummy with per-row norm = embed_scale so re-normalisation is a no-op.
        dummy_base = torch.randn(1, SEQ_LEN, HIDDEN_DIM)
        dummy_norms = dummy_base.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        dummy_embeds = dummy_base * (embed_scale / dummy_norms)  # norm = embed_scale

        anchor_embeds = torch.randn(1, ANCHOR_LEN, HIDDEN_DIM)
        mgr.get_model().model.embed_tokens.side_effect = [dummy_embeds, anchor_embeds]

        strategy = LatentStrategy(mgr, injection_gate=0.0)
        tensor = torch.randn(SEQ_LEN, HIDDEN_DIM)
        await strategy.decode(tensor)

        first_call = mgr.get_model().call_args_list[0]
        soft_prompt = first_call.kwargs["inputs_embeds"][0, :SEQ_LEN, :]
        assert torch.allclose(soft_prompt, dummy_embeds.squeeze(0), atol=1e-5), (
            "At gate=0 the soft-prompt must equal the dummy baseline"
        )

    async def test_decode_unit_gate_gives_hidden_matched(self, mgr):
        """At injection_gate=1 the soft-prompt must equal hidden_matched (the
        NormMatch-rescaled adapter output), since delta = hidden_matched - dummy
        and x_soft = dummy + 1.0 * delta = hidden_matched."""
        embed_scale = mgr.get_model().model.embed_tokens.weight.norm(dim=-1).mean()

        # Gate=1: dummy cancels out, x_soft = hidden_matched (already at embed_scale)
        dummy_embeds = torch.zeros(1, SEQ_LEN, HIDDEN_DIM)  # zero dummy
        anchor_embeds = torch.randn(1, ANCHOR_LEN, HIDDEN_DIM)
        mgr.get_model().model.embed_tokens.side_effect = [dummy_embeds, anchor_embeds]

        tensor = torch.randn(SEQ_LEN, HIDDEN_DIM)
        # Precompute the expected hidden_matched
        h = tensor.unsqueeze(0)
        norms = h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        expected = (h * (embed_scale / norms)).squeeze(0)

        strategy = LatentStrategy(mgr, injection_gate=1.0)
        await strategy.decode(tensor)

        first_call = mgr.get_model().call_args_list[0]
        soft_prompt = first_call.kwargs["inputs_embeds"][0, :SEQ_LEN, :]
        assert torch.allclose(soft_prompt, expected, atol=1e-4), (
            "At gate=1 the soft-prompt must equal hidden_matched"
        )

    async def test_decode_injection_gate_configurable(self, mgr):
        """LatentStrategy(mgr, injection_gate=X) must store and use that gate."""
        strategy = LatentStrategy(mgr, injection_gate=0.42)
        assert strategy._injection_gate == 0.42

    async def test_decode_default_injection_gate_is_0_3(self, mgr):
        """The default injection_gate must be 0.3.

        At 0.1 the adapter signal (10%) was too weak to overcome the EOS
        baseline's multilingual prior.  0.3 gives the adapter (30%) enough
        directional influence while the EOS baseline still provides manifold
        stability for the remaining 70%.
        """
        strategy = LatentStrategy(mgr)
        assert strategy._injection_gate == 0.3

    async def test_decode_prefix_pass_has_position_ids(self, strategy, mgr):
        """The prefix pass (inputs_embeds call) must include explicit position_ids."""
        tensor = torch.randn(SEQ_LEN, HIDDEN_DIM)
        await strategy.decode(tensor)
        first_call = mgr.get_model().call_args_list[0]
        assert "position_ids" in first_call.kwargs, (
            "Prefix pass must include position_ids for RoPE alignment"
        )

    async def test_decode_prefix_position_ids_span_full_prefix(self, strategy, mgr):
        """prefix position_ids must be arange(K+L) — one ID per embedded token."""
        tensor = torch.randn(SEQ_LEN, HIDDEN_DIM)
        prefix_len = SEQ_LEN + ANCHOR_LEN
        await strategy.decode(tensor)
        first_call = mgr.get_model().call_args_list[0]
        pos_ids = first_call.kwargs["position_ids"]
        assert pos_ids.shape == (1, prefix_len), (
            f"Expected position_ids shape (1, {prefix_len}), got {tuple(pos_ids.shape)}"
        )
        expected = torch.arange(prefix_len).unsqueeze(0)
        assert torch.equal(pos_ids.cpu(), expected), (
            "position_ids must be 0, 1, 2, ..., prefix_len-1"
        )

    async def test_decode_loop_calls_have_position_ids(self, strategy, mgr):
        """Every autoregressive loop call must include explicit position_ids."""
        tensor = torch.randn(SEQ_LEN, HIDDEN_DIM)
        await strategy.decode(tensor)
        loop_calls = mgr.get_model().call_args_list[1:]
        assert len(loop_calls) >= 1
        for i, c in enumerate(loop_calls):
            assert "position_ids" in c.kwargs, (
                f"Loop call {i} missing position_ids"
            )

    async def test_decode_first_loop_position_id_starts_at_prefix_len(
        self, strategy, mgr
    ):
        """First loop call position_ids must be [[K+L]] (immediately after prefix)."""
        tensor = torch.randn(SEQ_LEN, HIDDEN_DIM)
        prefix_len = SEQ_LEN + ANCHOR_LEN
        await strategy.decode(tensor)
        first_loop = mgr.get_model().call_args_list[1]
        pos_ids = first_loop.kwargs["position_ids"]
        assert pos_ids.tolist() == [[prefix_len]], (
            f"First loop position_ids must be [[{prefix_len}]], got {pos_ids.tolist()}"
        )

    async def test_decode_loop_position_ids_increment_each_step(self, strategy, mgr):
        """Loop position IDs must increment by 1 each step."""
        tensor = torch.randn(SEQ_LEN, HIDDEN_DIM)
        prefix_len = SEQ_LEN + ANCHOR_LEN
        await strategy.decode(tensor)
        loop_calls = mgr.get_model().call_args_list[1:]
        for step, c in enumerate(loop_calls):
            expected_pos = prefix_len + step
            actual = c.kwargs["position_ids"].item()
            assert actual == expected_pos, (
                f"Loop step {step}: expected position_id={expected_pos}, got {actual}"
            )


# ---------------------------------------------------------------------------
# Interface contracts
# ---------------------------------------------------------------------------

class TestLatentStrategyInterface:
    def test_is_subclass_of_thinking_strategy(self):
        from libucks.thinking.base import ThinkingStrategy
        assert issubclass(LatentStrategy, ThinkingStrategy)

    def test_instantiation_with_manager_stores_it(self, mgr):
        strategy = LatentStrategy(mgr)
        assert strategy._mgr is mgr

    def test_instantiation_without_manager_is_valid(self):
        strategy = LatentStrategy()
        assert strategy._mgr is None


# ---------------------------------------------------------------------------
# decode() — Sampling: temperature, top-p, repetition penalty
# ---------------------------------------------------------------------------

class TestLatentStrategySampling:
    """Tests for temperature sampling + repetition penalty to break greedy attractor.

    The root cause of "1. 1. 1." repetition is argmax (T=0) creating a
    deterministic 3-token cycle.  Fix: replace argmax with multinomial sampling
    gated by temperature / top-p / repetition-penalty.

    Written BEFORE the implementation (TDD).  All tests FAIL until
    _sample_next_token() is added and decode() uses it.
    """

    # --- Constructor / defaults ---

    def test_constructor_stores_temperature(self, mgr):
        s = LatentStrategy(mgr, temperature=0.5)
        assert s._temperature == 0.5

    def test_constructor_stores_top_p(self, mgr):
        s = LatentStrategy(mgr, top_p=0.8)
        assert s._top_p == 0.8

    def test_constructor_stores_repetition_penalty(self, mgr):
        s = LatentStrategy(mgr, repetition_penalty=1.5)
        assert s._repetition_penalty == 1.5

    def test_default_temperature_is_0_7(self):
        s = LatentStrategy()
        assert s._temperature == 0.7

    def test_default_top_p_is_0_9(self):
        s = LatentStrategy()
        assert s._top_p == 0.9

    def test_default_repetition_penalty_is_1_3(self):
        s = LatentStrategy()
        assert s._repetition_penalty == 1.3

    # --- _sample_next_token() shape / range ---

    def test_sample_next_token_returns_shape_1_tensor(self):
        s = LatentStrategy()
        logits = torch.randn(VOCAB_SIZE)
        result = s._sample_next_token(logits, [])
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1,)

    def test_sample_next_token_result_in_vocab_range(self):
        s = LatentStrategy()
        logits = torch.randn(VOCAB_SIZE)
        for _ in range(20):
            tok = s._sample_next_token(logits, []).item()
            assert 0 <= tok < VOCAB_SIZE

    # --- Repetition penalty ---

    def test_repetition_penalty_suppresses_seen_tokens(self):
        """With extreme penalty=1000.0, previously generated tokens are never sampled.

        Background tokens are set to 1.0 so that after token 5 is divided by 1000
        (0.01 << 1.0), the 31999 background tokens dominate and token 5 is
        effectively impossible to sample.
        """
        s = LatentStrategy(temperature=1.0, top_p=1.0, repetition_penalty=1000.0)
        logits = torch.full((VOCAB_SIZE,), 1.0)   # background at 1.0
        logits[5] = 10.0   # strong prior on token 5 — divided to 0.01 after penalty
        generated = [5]
        results = {s._sample_next_token(logits, generated).item() for _ in range(50)}
        assert 5 not in results, (
            f"Token 5 must be suppressed by extreme repetition penalty; got {results}"
        )

    # --- Temperature ---

    def test_near_zero_temperature_selects_argmax(self):
        """At near-zero temperature, sampling must deterministically pick the argmax."""
        s = LatentStrategy(temperature=1e-6, top_p=1.0, repetition_penalty=1.0)
        logits = torch.full((VOCAB_SIZE,), -1.0)
        logits[42] = 100.0
        for _ in range(10):
            assert s._sample_next_token(logits, []).item() == 42

    # --- Edge cases ---

    def test_sample_next_token_empty_generated_ids_does_not_crash(self):
        s = LatentStrategy()
        logits = torch.randn(VOCAB_SIZE)
        result = s._sample_next_token(logits, [])
        assert isinstance(result, torch.Tensor)

    # --- Integration: decode() uses sampling ---

    async def test_decode_uses_sampling_not_argmax(self, mgr):
        """decode() must delegate to _sample_next_token for every generated token."""
        from unittest.mock import patch
        s = LatentStrategy(mgr)
        tensor = torch.randn(SEQ_LEN, HIDDEN_DIM)

        call_count = 0
        original = s._sample_next_token

        def counting_sample(logits, generated_ids):
            nonlocal call_count
            call_count += 1
            return original(logits, generated_ids)

        with patch.object(s, "_sample_next_token", side_effect=counting_sample):
            await s.decode(tensor)

        assert call_count >= 1, (
            "_sample_next_token must be called at least once during decode()"
        )


# ---------------------------------------------------------------------------
# receive() — Interlat-Lite Base receiver path (Phase 12.7)
# ---------------------------------------------------------------------------

def _make_base_model_mock() -> MagicMock:
    """Mock Base model for receive() tests — no embed_tokens.weight needed."""
    m = MagicMock()
    out = MagicMock()
    logits = torch.full((1, 1, VOCAB_SIZE), -1.0)
    logits[0, 0, 100] = 10.0  # always predicts token 100 (non-EOS)
    out.logits = logits
    out.past_key_values = MagicMock()
    m.return_value = out
    m.side_effect = None
    return m


def _make_mock_mgr_with_base(seq_len: int = SEQ_LEN) -> MagicMock:
    """ModelManager mock that has both instruct and base model/tokenizer."""
    mgr = _make_mock_mgr(seq_len)
    # Base tokenizer
    base_tok = MagicMock()
    base_tok.eos_token_id = 2
    base_tok.decode.return_value = "Fetches the user profile from the database."
    mgr.get_base_tokenizer.return_value = base_tok
    # Base model
    base_model = _make_base_model_mock()
    mgr.get_base_model.return_value = base_model
    return mgr


class TestLatentStrategyReceive:
    """Tests for LatentStrategy.receive() — the Interlat-Lite decode path.

    receive() is the ONLY authorised path for decoding in the latent pipeline.
    It must use the Base model (not Instruct) and must NOT apply NormMatch or
    Residual Anchoring — the Base model has been LoRA-trained to accept raw
    framed latents directly.
    """

    @pytest.fixture
    def mgr_with_base(self) -> MagicMock:
        return _make_mock_mgr_with_base()

    @pytest.fixture
    def strategy_with_base(self, mgr_with_base: MagicMock) -> LatentStrategy:
        return LatentStrategy(mgr_with_base)

    @pytest.fixture
    def framed_latent(self) -> torch.Tensor:
        """Shape (K+2, D) — includes bop/eop rows."""
        K = 8
        return torch.randn(K + 2, HIDDEN_DIM)

    async def test_receive_returns_str(self, strategy_with_base, framed_latent):
        """receive() must return a plain string."""
        result = await strategy_with_base.receive(framed_latent)
        assert isinstance(result, str)

    async def test_receive_uses_base_model_not_instruct(
        self, strategy_with_base, mgr_with_base, framed_latent
    ):
        """receive() must call get_base_model(), not get_model()."""
        await strategy_with_base.receive(framed_latent)
        mgr_with_base.get_base_model.assert_called()
        mgr_with_base.get_model.assert_not_called()

    async def test_receive_uses_base_tokenizer_not_instruct(
        self, strategy_with_base, mgr_with_base, framed_latent
    ):
        """receive() must call get_base_tokenizer(), not get_tokenizer()."""
        await strategy_with_base.receive(framed_latent)
        mgr_with_base.get_base_tokenizer.assert_called()
        mgr_with_base.get_tokenizer.assert_not_called()

    async def test_receive_does_not_use_normatch(
        self, strategy_with_base, mgr_with_base, framed_latent
    ):
        """receive() must NOT call embed_tokens — NormMatch is absent.

        decode() calls model.model.embed_tokens() twice (dummy baseline + anchor).
        receive() must call it zero times.
        """
        await strategy_with_base.receive(framed_latent)
        base_model = mgr_with_base.get_base_model()
        # If NormMatch were present, base_model.model.embed_tokens() would be invoked.
        # MagicMock tracks child-attribute call counts:
        embed_tokens_call_count = base_model.model.embed_tokens.call_count
        assert embed_tokens_call_count == 0, (
            f"receive() called embed_tokens {embed_tokens_call_count} time(s); "
            "NormMatch must be absent from the Base receiver path"
        )

    async def test_receive_calls_base_model_with_inputs_embeds(
        self, strategy_with_base, mgr_with_base, framed_latent
    ):
        """receive() must inject the framed tensor as inputs_embeds."""
        await strategy_with_base.receive(framed_latent)
        base_model = mgr_with_base.get_base_model()
        # At least one call must use inputs_embeds
        calls_with_embeds = [
            c for c in base_model.call_args_list
            if "inputs_embeds" in c.kwargs
        ]
        assert len(calls_with_embeds) >= 1, (
            "receive() must pass inputs_embeds to the base model"
        )

    async def test_receive_accepts_2d_framed_latent(
        self, strategy_with_base, framed_latent
    ):
        """framed_latent of shape (K+2, D) must work — no batch dim needed."""
        assert framed_latent.dim() == 2
        result = await strategy_with_base.receive(framed_latent)
        assert isinstance(result, str)

    @pytest.mark.skip(reason="Requires GPU + trained LoRA weights — integration only")
    async def test_receive_returns_coherent_string_after_training(self):
        """After training, receive() must produce alphabetic parseable English.

        This test is skipped in CI — run manually after LoRA fine-tuning completes.
        """
        pass
