"""Chunked cache building must equal the single-forward version.

WHY IT EXISTS. CM-B.0j measured that ONE 4,599-token forward permanently caches
4.5 GB that `torch.mps.empty_cache()` will not return, so the real memory
requirement was 12.3 GB against 7.0 GB of actual tensors. That is why the ceiling
control took four launch attempts on a 16 GB machine.

WHY IT IS NOW MANDATORY, not merely nice. Query-aware selection (kv_attn) needs
attention weights, and transformers 5.4.0 with SDPA returns `attentions=()` — it
does NOT fall back to eager. So the model must be loaded with
`attn_implementation='eager'`, and eager attention over 4,599 tokens in one forward
materialises a (1, heads, 4599, 4599) matrix per layer. Chunking is what makes
eager affordable: at 512 tokens the largest matrix is ~9x smaller in the sequence
dimension it squares.

WHAT IS AND IS NOT GUARANTEED. Under causal attention a token's K/V depend only on
itself and earlier tokens, so chunking computes the same values. It does NOT compute
them with the same matmul shapes, so bf16/fp32 accumulation order differs and the
result is close but not bitwise identical. These tests assert closeness with a
tolerance, and the caller must accept that a chunked cache can shift a greedy decode
on a near-tie. Anything asserting bitwise equality here would be wrong.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

HF = Path.home() / ".cache/huggingface/hub"
pytestmark = pytest.mark.skipif(
    not (HF / "models--gpt2").exists(),
    reason="gpt2 not in the local HF cache",
)


@pytest.fixture(scope="module")
def gpt2():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    m = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32).eval()
    return m, AutoTokenizer.from_pretrained("gpt2")


TEXT = " ".join(f"line {i} defines value_{i} as {i * 7}." for i in range(60))


class TestSignature:
    def test_chunk_tokens_is_opt_in(self):
        """Default must stay None so the ~10 existing callers are untouched."""
        import inspect

        from libucks.cache_augmentation.kv_extract import extract_bucket_kv

        p = inspect.signature(extract_bucket_kv).parameters
        assert "chunk_tokens" in p
        assert p["chunk_tokens"].default is None


class TestEquivalence:
    def _both(self, gpt2, chunk):
        from libucks.cache_augmentation.kv_extract import extract_bucket_kv

        m, tok = gpt2
        a = extract_bucket_kv(m, tok, TEXT, max_tokens=256)
        b = extract_bucket_kv(m, tok, TEXT, max_tokens=256, chunk_tokens=chunk)
        return a, b

    def test_same_shape_and_layer_count(self, gpt2):
        a, b = self._both(gpt2, 64)
        assert set(a) == set(b)
        for k in a:
            assert a[k].shape == b[k].shape, k

    def test_same_seq_len_metadata(self, gpt2):
        a, b = self._both(gpt2, 64)
        assert int(a["_meta_seq_len"]) == int(b["_meta_seq_len"])
        assert int(a["_meta_n_layers"]) == int(b["_meta_n_layers"])

    @pytest.mark.parametrize("chunk", [32, 64, 128])
    def test_values_match_within_float_tolerance(self, gpt2, chunk):
        """Close, not bitwise — different matmul shapes accumulate differently."""
        a, b = self._both(gpt2, chunk)
        worst, where = 0.0, ""
        for k in a:
            if k.startswith("_meta"):
                continue
            d = (a[k].float() - b[k].float()).abs().max().item()
            if d > worst:
                worst, where = d, k
        assert worst < 1e-3, f"max abs diff {worst} at {where}"

    def test_chunk_larger_than_input_is_a_single_forward(self, gpt2):
        a, b = self._both(gpt2, 10_000)
        for k in a:
            if not k.startswith("_meta"):
                assert torch.equal(a[k], b[k]), (
                    f"{k}: one chunk covering everything must be the plain path"
                )

    def test_chunk_boundary_does_not_truncate(self, gpt2):
        """A boundary that does not divide the length evenly must still cover
        every token — an off-by-one here silently shortens the cache, which is
        exactly the class of bug that has bitten this project three times."""
        from libucks.cache_augmentation.kv_extract import extract_bucket_kv

        m, tok = gpt2
        full = extract_bucket_kv(m, tok, TEXT, max_tokens=200)
        n = int(full["_meta_seq_len"])
        # 199 against 200 tokens is the load-bearing case: it is the only one that
        # leaves a 1-token tail, and a 1-token continuation forward SIGBUSes GPT-2
        # on transformers 5.4.0 (verified in a minimal repro with no libucks code;
        # 198+2, 196+4 and 150+50 all pass). extract_bucket_kv folds such a tail
        # into the previous segment, so this must pass rather than crash.
        for chunk in (7, 33, 199, 200, 201):
            got = extract_bucket_kv(m, tok, TEXT, max_tokens=200,
                                    chunk_tokens=chunk)
            assert int(got["_meta_seq_len"]) == n, f"chunk={chunk} lost tokens"
            assert got["layer_0_K"].shape[2] == n, f"chunk={chunk} short cache"


class TestRejectsNonsense:
    def test_zero_or_negative_chunk_raises(self, gpt2):
        from libucks.cache_augmentation.kv_extract import extract_bucket_kv

        m, tok = gpt2
        for bad in (0, -1):
            with pytest.raises(ValueError, match="chunk_tokens"):
                extract_bucket_kv(m, tok, TEXT, max_tokens=64, chunk_tokens=bad)
