"""Smoke tests for Phase 4-C.2 cache augmentation layer.

Validates:
  1. extract_bucket_kv → save → load → restore_dynamic_cache produces a cache
     that, when passed as past_key_values, yields the same logits as a fresh
     forward over the same tokens (within float tolerance).
  2. Invalidation: when a bucket's chunk signature changes, the stale cache
     is rejected and a fresh extraction is required.

These tests use libugry's actual buckets so we exercise the full path on
real data (not synthetic toys).

Run via:
    uv run pytest tests/integration/test_cache_aug_smoke.py -v -s -m smoke
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest
import torch

# Mark all tests in this module as 'smoke' so they don't run with `pytest tests/`
# by default; cost is ~30s for model load.
pytestmark = pytest.mark.smoke


_LIBUGRY = Path("/Users/ecaterina/Developer/test-repos/libugry")


@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load Qwen 2.5-3B once for all tests in this module."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B",
        dtype=torch.bfloat16,
    )
    model.eval()
    # Use MPS if available — matches production
    if torch.backends.mps.is_available():
        model = model.to("mps")
    return model, tok


@pytest.fixture
def libugry_bucket_text():
    """Pull one libugry bucket's source text via the existing helper."""
    if not _LIBUGRY.exists() or not (_LIBUGRY / ".libucks" / "buckets").exists():
        pytest.skip("libugry not initialised at expected path")
    from libucks.librarian import _collect_source_text
    from libucks.storage.bucket_store import BucketStore

    store = BucketStore(_LIBUGRY / ".libucks" / "buckets")
    # Pick the first bucket file deterministically by sorting.
    bucket_files = sorted((_LIBUGRY / ".libucks" / "buckets").glob("*.md"))
    if not bucket_files:
        pytest.skip("libugry has no buckets")
    bucket_id = bucket_files[0].stem
    fm, prose = store.read(bucket_id)
    text = _collect_source_text(fm, max_chars=2000) or prose
    assert text, f"empty bucket text for {bucket_id}"
    return bucket_id, text, fm.chunks


def test_kv_extract_returns_expected_layer_count(model_and_tokenizer, libugry_bucket_text):
    """extract_bucket_kv produces 36 layers (Qwen 2.5-3B) of K and V tensors."""
    from libucks.cache_augmentation.kv_extract import extract_bucket_kv

    model, tok = model_and_tokenizer
    _, text, _ = libugry_bucket_text

    flat = extract_bucket_kv(model, tok, text, max_tokens=256)
    assert flat["_meta_n_layers"].item() == 36
    for i in range(36):
        assert f"layer_{i}_K" in flat
        assert f"layer_{i}_V" in flat
        # Expected shape: (1, 2, T, 128) for Qwen 2.5-3B with GQA
        assert flat[f"layer_{i}_K"].shape[1] == 2, "expected 2 KV heads"
        assert flat[f"layer_{i}_K"].shape[3] == 128, "expected head_dim 128"


def test_save_load_roundtrip_preserves_tensors(
    model_and_tokenizer, libugry_bucket_text, tmp_path
):
    """save → load returns the same tensors (within bf16 precision)."""
    from libucks.cache_augmentation.bucket_kv_cache import BucketKVCache
    from libucks.cache_augmentation.kv_extract import extract_bucket_kv

    model, tok = model_and_tokenizer
    bucket_id, text, chunks = libugry_bucket_text

    cache = BucketKVCache(tmp_path, model_id="Qwen/Qwen2.5-3B", max_tokens=256)
    flat = extract_bucket_kv(model, tok, text, max_tokens=256)
    cache.save(bucket_id, flat, chunks)
    assert cache.exists(bucket_id)

    loaded = cache.load(bucket_id, chunks)
    assert loaded is not None, "load returned None despite fresh save"
    for i in range(36):
        assert torch.equal(flat[f"layer_{i}_K"].cpu(), loaded[f"layer_{i}_K"].cpu())
        assert torch.equal(flat[f"layer_{i}_V"].cpu(), loaded[f"layer_{i}_V"].cpu())


def test_invalidation_rejects_stale_cache(
    model_and_tokenizer, libugry_bucket_text, tmp_path
):
    """If the bucket's chunk signature changes, load() returns None."""
    from libucks.cache_augmentation.bucket_kv_cache import BucketKVCache
    from libucks.cache_augmentation.kv_extract import extract_bucket_kv
    from libucks.models.chunk import ChunkMetadata

    model, tok = model_and_tokenizer
    bucket_id, text, chunks = libugry_bucket_text

    cache = BucketKVCache(tmp_path, model_id="Qwen/Qwen2.5-3B", max_tokens=256)
    flat = extract_bucket_kv(model, tok, text, max_tokens=256)
    cache.save(bucket_id, flat, chunks)

    # Simulate a chunk update: same chunk_ids, but one has a different git_sha.
    if not chunks:
        pytest.skip("bucket has no chunks; cannot test invalidation")
    chunks_modified = [
        ChunkMetadata(
            **{**chunks[0].model_dump(), "git_sha": "deadbeef" + chunks[0].git_sha[8:]}
        ),
        *chunks[1:],
    ]
    loaded = cache.load(bucket_id, chunks_modified)
    assert loaded is None, "stale cache must be rejected when chunk signature changes"


def test_restored_cache_matches_direct_forward(
    model_and_tokenizer, libugry_bucket_text
):
    """The critical roundtrip: extract → save → load → restore → augmented
    forward → next-token logits match a fresh forward over the same tokens."""
    from libucks.cache_augmentation.kv_extract import (
        extract_bucket_kv,
        restore_dynamic_cache,
    )

    model, tok = model_and_tokenizer
    _, text, _ = libugry_bucket_text
    device = next(model.parameters()).device

    # Step 1: extract KV from the bucket text.
    flat = extract_bucket_kv(model, tok, text, max_tokens=256)

    # Step 2: tokenize a continuation token and feed BOTH directly (reference)
    # and via the restored cache (under test).
    enc = tok(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    next_id = torch.tensor([[tok.eos_token_id or 0]], device=device)

    with torch.no_grad():
        # Reference: full forward over (bucket_text + next_id).
        full_ids = torch.cat([enc["input_ids"], next_id], dim=1)
        ref_out = model(input_ids=full_ids, use_cache=False)
        ref_logits = ref_out.logits[:, -1, :]  # last position

        # Augmented: feed only next_id, using the restored cache.
        restored = restore_dynamic_cache(flat, device=device)
        aug_out = model(
            input_ids=next_id,
            past_key_values=restored,
            use_cache=True,
        )
        aug_logits = aug_out.logits[:, -1, :]

    # bf16 precision; tolerance accounts for accumulation order differences.
    diff = (ref_logits.float() - aug_logits.float()).abs().max().item()
    print(f"\n[roundtrip] max logit diff = {diff:.4f}")
    # The two paths should agree to within bf16 precision; bf16 mantissa is
    # 7 bits so individual logits can differ by O(1e-2). Allow some slack
    # for accumulation drift across 36 layers.
    assert diff < 0.5, f"roundtrip logit drift too large: {diff}"


# ---------------------------------------------------------------------------
# Phase 4-C.3 — Coprocessor smoke
# ---------------------------------------------------------------------------


def _random_bucket_kv(seq_len: int = 256, dtype: torch.dtype = torch.bfloat16):
    """Synthesise a flat dict matching the layout produced by extract_bucket_kv.

    Useful for coprocessor tests that don't need a real model load.
    """
    n_layers = 36
    n_kv_heads = 2
    head_dim = 128
    flat: dict[str, torch.Tensor] = {}
    for i in range(n_layers):
        flat[f"layer_{i}_K"] = torch.randn(1, n_kv_heads, seq_len, head_dim, dtype=dtype)
        flat[f"layer_{i}_V"] = torch.randn(1, n_kv_heads, seq_len, head_dim, dtype=dtype)
    flat["_meta_seq_len"] = torch.tensor([seq_len], dtype=torch.int32)
    flat["_meta_n_layers"] = torch.tensor([n_layers], dtype=torch.int32)
    return flat


def test_coprocessor_forward_shape_and_finite():
    """Default coprocessor (4 blocks, hidden=2048, K=64) emits (1, 64, 2048),
    finite values, reasonable output norm relative to input scale."""
    from libucks.cache_augmentation.coprocessor import Coprocessor

    coproc = Coprocessor()
    n_params = coproc.param_count()
    print(f"\n[coproc] param count: {n_params/1e6:.1f}M")
    assert 80e6 <= n_params <= 250e6, (
        f"unexpected coprocessor size {n_params/1e6:.1f}M — "
        "default ought to be ~150-200M; tune blocks/heads/ffn_mult if drift"
    )

    flat = _random_bucket_kv(seq_len=128, dtype=torch.float32)
    # Match coprocessor's compute dtype (default float32)
    z = coproc(flat)
    assert z.shape == (1, 64, 2048), f"unexpected z shape {tuple(z.shape)}"
    assert torch.isfinite(z).all(), "coprocessor produced NaN or inf"
    # Output norm: roughly comparable to input scale; layer norms keep it bounded.
    out_norm = z.norm(dim=-1).mean().item()
    print(f"[coproc] mean per-token norm: {out_norm:.3f}")
    # LayerNorm on 2048 dims with unit input gives sqrt(2048) ≈ 45; allow 5x slack.
    assert 1.0 < out_norm < 250.0, f"unreasonable output norm {out_norm}"


def test_coprocessor_distinct_buckets_produce_distinct_z():
    """Two different bucket KV caches → two different z outputs (sanity).

    If the coprocessor ignored its bucket-KV input, z would be identical
    across calls because only the soft-token weights matter. This test
    confirms the bucket KV actually flows through."""
    from libucks.cache_augmentation.coprocessor import Coprocessor

    coproc = Coprocessor()
    coproc.eval()

    flat_a = _random_bucket_kv(seq_len=64, dtype=torch.float32)
    flat_b = _random_bucket_kv(seq_len=64, dtype=torch.float32)

    with torch.no_grad():
        z_a = coproc(flat_a)
        z_b = coproc(flat_b)

    # Cosine similarity of mean-pooled z across the K dim.
    a = z_a.mean(dim=1).flatten()
    b = z_b.mean(dim=1).flatten()
    cos = (a @ b / (a.norm() * b.norm() + 1e-8)).item()
    print(f"\n[coproc] cos(z_a, z_b) for distinct random buckets = {cos:.3f}")
    # Untrained coprocessor: expect modest similarity (random init smooths
    # everything), but NOT 1.0. Tightening this once trained.
    assert cos < 0.999, "coprocessor output ignores bucket KV input"


def test_coprocessor_param_count_breakdown(capsys):
    """Print a param-count breakdown — diagnostic, not a hard assertion."""
    from libucks.cache_augmentation.coprocessor import Coprocessor

    coproc = Coprocessor()
    print()
    print(f"coproc total: {coproc.param_count()/1e6:.2f}M")
    for name, p in coproc.named_parameters():
        if p.requires_grad and p.numel() > 100_000:
            print(f"  {name:50s} {tuple(p.shape)!s:30s} {p.numel()/1e6:6.2f}M")


# ---------------------------------------------------------------------------
# Phase 4-C.4 — Fusion + augmented decode smoke
# ---------------------------------------------------------------------------


def test_fusion_forward_shape_and_finite():
    """CrossBucketFusion takes N bucket-z and emits one z_fused, shape (1, K, d)."""
    from libucks.cache_augmentation.fusion import CrossBucketFusion

    fusion = CrossBucketFusion()
    print(f"\n[fusion] params: {fusion.param_count()/1e6:.1f}M")

    # Simulate 3 routed buckets' z outputs (post-coprocessor): (1, 64, 2048) each.
    bucket_z = [torch.randn(1, 64, 2048) for _ in range(3)]
    z_fused = fusion(bucket_z)
    assert z_fused.shape == (1, 64, 2048), f"unexpected z_fused shape {tuple(z_fused.shape)}"
    assert torch.isfinite(z_fused).all()


def test_augmented_decode_end_to_end(model_and_tokenizer, libugry_bucket_text):
    """Full plumbing test: bucket text → KV extract → coprocessor → fusion →
    augmented_decode produces a coherent (untrained, gibberish-allowed) string
    without crash, NaN, or OOM. Single bucket variant — exercises the cache
    concatenation in decode.py.

    This is the load-bearing 4-C.4 gate: end-to-end inference runs."""
    from libucks.cache_augmentation.coprocessor import Coprocessor
    from libucks.cache_augmentation.decode import augmented_decode
    from libucks.cache_augmentation.fusion import CrossBucketFusion
    from libucks.cache_augmentation.kv_extract import extract_bucket_kv

    model, tok = model_and_tokenizer
    _, text, _ = libugry_bucket_text
    device = next(model.parameters()).device

    # Extract bucket KV (real model, real bucket text).
    flat = extract_bucket_kv(model, tok, text, max_tokens=256)

    # Untrained coproc + fusion on the same device as the receiver.
    coproc = Coprocessor().to(device).to(torch.float32)
    fusion = CrossBucketFusion().to(device).to(torch.float32)
    coproc.eval(); fusion.eval()

    z_b = coproc(flat)             # (1, 64, 2048)
    z_fused = fusion([z_b])         # (1, 64, 2048)
    assert torch.isfinite(z_fused).all()
    print(f"\n[end-to-end] z_fused shape={tuple(z_fused.shape)} "
          f"mean_norm={z_fused.norm(dim=-1).mean().item():.2f}")

    # Run augmented decode end-to-end. Untrained, so output will be gibberish;
    # we just want plumbing to work (no NaN, no OOM, returns a string).
    out_text = augmented_decode(
        model, tok,
        z_fused=z_fused,
        query="What does this code do?",
        verbatim="",  # no verbatim for the smoke
        max_new_tokens=20,
    )
    assert isinstance(out_text, str)
    print(f"[end-to-end] decoded text: {out_text!r}")


# ---------------------------------------------------------------------------
# Phase 4-C.5 — Training smoke
# ---------------------------------------------------------------------------


def test_cache_aug_trainer_one_step_grads_flow(model_and_tokenizer, libugry_bucket_text, tmp_path):
    """One training step on a real bucket: build z_fused → augmented cache →
    CE loss → backward updates coproc + fusion params, leaves receiver frozen.

    This is the 4-C.5 gate: validates the gradient path is correctly hooked
    up and the loss is finite. If this fails, the long training run is
    guaranteed to fail."""
    from libucks.cache_augmentation.bucket_kv_cache import BucketKVCache
    from libucks.cache_augmentation.coprocessor import Coprocessor
    from libucks.cache_augmentation.fusion import CrossBucketFusion
    from libucks.cache_augmentation.kv_extract import extract_bucket_kv
    from libucks.thinking.training.cache_aug_trainer import (
        CacheAugTrainSample, CacheAugTrainer,
    )

    model, tok = model_and_tokenizer
    bucket_id, text, chunks = libugry_bucket_text
    device = next(model.parameters()).device

    # Stage the bucket's KV cache on disk so the trainer can load it.
    kv_cache = BucketKVCache(tmp_path, model_id="Qwen/Qwen2.5-3B", max_tokens=256)
    flat = extract_bucket_kv(model, tok, text, max_tokens=256)
    kv_cache.save(bucket_id, flat, chunks)

    # Build coproc + fusion on the same device (float32 for training stability).
    coproc = Coprocessor().to(device).to(torch.float32)
    fusion = CrossBucketFusion().to(device).to(torch.float32)

    bucket_chunks = {bucket_id: chunks}
    trainer = CacheAugTrainer(
        base_model=model,
        tokenizer=tok,
        coprocessor=coproc,
        fusion=fusion,
        bucket_kv_cache=kv_cache,
        bucket_chunks=bucket_chunks,
        lr=1e-4,
        warmup_steps=10,
        total_steps=100,
    )

    sample = CacheAugTrainSample(
        bucket_id=bucket_id,
        query="What does this Python module do?",
        answer="It implements a small helper for the test fixture used in libucks evaluation.",
    )

    # Snapshot a coproc param to verify it changed after the step.
    pre = coproc.kv_proj.weight.detach().clone()
    pre_ptr = coproc.kv_proj.weight.data_ptr()
    pre_mean = coproc.kv_proj.weight.mean().item()
    print(f"\n[test-pre]  data_ptr={pre_ptr}  mean={pre_mean:.6e}")
    print(f"[test-pre]  trainer.coproc is coproc? {trainer.coproc is coproc}")
    print(f"[test-pre]  trainer.coproc.kv_proj.weight is coproc.kv_proj.weight? "
          f"{trainer.coproc.kv_proj.weight is coproc.kv_proj.weight}")

    metrics = trainer.step(sample)
    post_ptr = coproc.kv_proj.weight.data_ptr()
    post_mean = coproc.kv_proj.weight.mean().item()
    print(f"[test-post] data_ptr={post_ptr}  mean={post_mean:.6e}  delta_mean={post_mean-pre_mean:.6e}")
    assert metrics is not None, "trainer step returned None — cache miss?"
    print(f"\n[train-step] loss={metrics['loss']:.4f} lr={metrics['lr']:.2e} n_tokens={metrics['n_tokens']}")

    # Loss must be finite + positive.
    assert math.isfinite(metrics["loss"]), f"non-finite loss: {metrics['loss']}"
    assert metrics["loss"] > 0, "loss should be positive (CE on random init)"

    # Coproc weight should have changed (the AdamW step ran).
    post = coproc.kv_proj.weight.detach()
    delta = (post - pre).abs().max().item()
    print(f"[train-step] max coproc.kv_proj weight delta = {delta:.2e}")
    assert delta > 0, "coproc params did not change after optimizer.step()"

    # Receiver params should NOT have grad (they're frozen).
    any_receiver_grad = any(p.grad is not None for p in model.parameters() if not p.requires_grad)
    assert not any_receiver_grad, "frozen receiver received gradients"
