# Plan — Phase 4-C: Cache augmentation prototype on echoswarm

## Status

Phase 4-A: PoC closeout DONE. Locked baseline: **hybrid 19.5 ± 1.7 / 30**
on libugry (4-run mean, verbatim-only configuration, MPS, bf16). The
routing-layer chunk rerank from Phase 3-B is deprecated; only the
in-bucket verbatim rerank is kept.

Phase 4-B: paper synthesis DONE. Key implementation decisions locked
(see [[project-paper-synthesis]]):
- Architecture = DeepMind cache augmentation (2412.17747).
- Training curriculum = Token Assorted's randomized AR replacement.
- Coprocessor on KV cache, NOT last-layer activations (DeepMind
  ablation: 26.76% vs 23.20% GSM8K).
- K=64 latent embeddings (validated by their ablation).
- Receiver frozen (drop the LoRA on the receiver).
- Verbatim channel + in-bucket chunk rerank: KEEP.
- Hierarchical multi-bucket fusion = libucks's research novelty.

Storage math (NEW): Qwen 2.5-3B uses GQA with only 2 KV heads, not 16.
Per-token KV cache = 36 layers × 2 KV heads × 128 head_dim × 2 (K+V)
× 2 bytes (bf16) = **37 KB/token**. A 500-token bucket → 18 MB.
30 buckets → ~540 MB. Far more practical than I had estimated.

## Goals

Build cache augmentation as a NEW eval mode (`cache_aug`) alongside
the existing 4 paths. A/B test against `hybrid` on a fresh repo
(echoswarm). Decision gate: whether to migrate, dual-track, or
future-work.

### Quantitative gates

After eval on **echoswarm + libugry cross-validation**:

| outcome | criterion (echoswarm hybrid as baseline) | action |
|---|---|---|
| Strong win | `cache_aug` ≥ `hybrid` + 3 grounding (above noise) AND `cache_aug` latent-alone variant ≥ no_context + 5 (latent claim restored) | Migrate (4-D-A). Cache aug becomes default. |
| Equal | `cache_aug` ≈ `hybrid` (±2 grounding) | Dual-track (4-D-B). Both modes shipped. |
| Loss | `cache_aug` < `hybrid` - 2 | Future-work (4-D-C). Phase 4-A numbers stay as headline. |

Architectural claim restoration (independent of grounding):
- `cache_aug` (no verbatim) ≥ no_context + 5 → the "latent communication
  alone is viable" claim is restored, separate from hybrid grounding.

## Architecture (locked decisions)

### High-level pipeline (per query)

```
Query q
  ↓
[A] embedder.embed(q) → q_emb
[B] CentralAgent.route(q_emb, top_k=3) → bucket_ids
[C] For each bucket b in bucket_ids:
        load BucketKVCache(b) from disk        # precomputed at indexing
        Coprocessor(cache_b, soft_tokens) → z_b   # (K=64, 2048)
[D] CrossBucketFusion(z_1, z_2, z_3) → z_fused    # (K=64, 2048)
[E] Frozen receiver forward over z_fused (no decode) → C_z   # KV cache for z_fused
[F] Frozen receiver forward over (verbatim + query) → C_input  # KV cache for plain context
[G] Concatenate C_input + C_z → C_aug
[H] Frozen receiver generate from C_aug
  ↓
answer text
```

### Component sizes

| component | size | trained? | notes |
|---|---|---|---|
| Frozen receiver | Qwen 2.5-3B (~6.4 GB bf16) | NO | DeepMind: frozen base is the design constraint |
| Per-bucket KV cache | ~18 MB/bucket on disk (bf16); ~9 MB int8 | precomputed at indexing | 36 layers × 2 KV heads × 128 head_dim |
| Coprocessor | Lightweight: 8 transformer blocks, hidden=2048, cross-attn to bucket KV → ~150-250M params | YES | Init: middle layers of Qwen-3B (layers 16-23). Lightweight to fit in MPS memory. |
| Cross-bucket fusion | 2-layer transformer (same as current CommunicationAdapter shape) | YES | Warm-start from existing `adapter.pt` weights. K=64 output queries preserved. |
| Soft token embeddings | (K=64, 2048) trainable | YES | Standard learned positional queries |

**Total trainable params**: ~150-250M (coprocessor) + ~20M (fusion +
soft tokens) = under 300M. Fits MPS comfortably with gradient
checkpointing.

**Total inference memory**: 6.4 GB (frozen) + ~500 MB (coprocessor in
bf16, only activations held during query) = ~7 GB. Headroom.

**Fallback if quality insufficient**: scale coprocessor to a full
Qwen-3B clone with LoRA-128 (per DeepMind's recipe). Documented in
the codebase, not implemented up-front. Decision deferred to 4-C.5
based on intermediate results.

### Multi-bucket fusion (libucks's novelty)

Per-bucket coprocessor outputs z_b at shape (K=64, 2048). Fusion
block takes [z_1, z_2, z_3] (shape (3, K, 2048) after stacking),
applies cross-bucket self-attention via learned K=64 output queries
(same pattern as current `CommunicationAdapter`), emits z_fused at
(K=64, 2048). Critically, this is the SAME architecture as the current
adapter — we're just changing what feeds it (per-bucket cache-aug
outputs instead of per-bucket Librarian Representations). Warm-start
weights from `adapter.pt` to bootstrap training.

### Training curriculum (Token Assorted-style)

For each training frame:
- Sample `text_ratio ∈ {0.0, 0.25, 0.5, 0.75, 1.0}` uniformly.
- Verbatim channel sees `text_ratio × max_chars` of source text.
- Cache aug channel always active.
- Loss = standard LM loss on the answer tokens (DeepMind recipe).

Replaces our current binary "50% hybrid-train" pattern. Token Assorted's
ablation (Table 4.4) shows this beats curriculum learning by 2-15 points
across model sizes.

### What we KEEP unchanged from Phase 4-A

- `CentralAgent.route()`: centroid-only (Phase 3-B routing-layer rerank
  remains deprecated).
- `ChunkRetriever`: still used for in-bucket chunk rerank in the
  verbatim channel.
- `BucketRegistry`, `BucketStore`, frontmatter format (extended with
  `kv_cache_path`).
- Self-evolving services: `NovelBucketService`, `MitosisService`,
  `MergingService`, `HealthMonitor`.
- `Translator` as sole decode point (faithfulness constraint).
- MCP surface, git-hook indexing, per-repo `.libucks/` layout.

### What we DROP

- The current `LoRA receiver` (trained on `q_proj, v_proj, o_proj` of
  Qwen-3B). DeepMind paper uses frozen receiver. The `lora_receiver.pt`
  file is preserved as the Phase 4-A baseline weights but not loaded
  by the `cache_aug` mode.
- `L_sep` ranking loss curriculum from `_train_lora_receiver`.
- Phase 3-A diagnostic paths (`hybrid_clean`, `latent_clean`) — done.

## Persistence (durable artifacts for the 3-week arc)

Two files in the repo, version-controlled, so the plan and progress
survive session boundaries:

- **`docs/phase-4c-plan.md`** — verbatim copy of THIS plan. Created
  at the start of 4-C.1. Refreshed only if the plan changes (with
  a "Plan revision" entry in the log noting why). When I (Claude)
  start a session, reading this file restores full project context
  without needing to crawl memory entries.
- **`docs/phase-4c-log.md`** — append-only progress log. After each
  sub-phase (4-C.1, 4-C.2, ...) completes, write a structured entry:
  date, sub-phase, what was built, decisions made, what worked, what
  broke, blockers, gate-pass verdict, time spent, what's next. Format
  is a markdown table-of-contents at top + sections below. Pattern:

  ```
  ## 4-C.1 — Infrastructure prep (2026-06-13 → 2026-06-15)

  **Status**: ✅ gate passed / ⚠ partial / ❌ blocked
  **Built**: <bullets>
  **Decided**: <bullets>
  **Issues**: <bullets>
  **Gate result**: <met / not met>; next gate is <X>
  **Time**: <Y working days>
  **Next**: 4-C.2 — KV cache extraction
  ```

This pattern is also lifted as a feedback memory entry so future
sessions on long projects default to it.

## Sub-phases (each with go/no-go gate)

### 4-C.1 — Infrastructure prep (~2 days)

Setup that's independent of training.

| step | output | gate |
|---|---|---|
| Init libucks on echoswarm: `libucks install-hooks` + initial commit replay | `.libucks/` populated with buckets, centroids, prose | ≥10 buckets formed; centroid quality reasonable by manual inspection |
| Hand-curate ~25 echoswarm fixtures in libugry_qa.json schema | `tests/eval/fixtures/echoswarm_qa.json` | All fixtures have grounded keywords; manual review |
| Wire echoswarm into eval harness (extend `_REPOS`) | Harness recognises echoswarm | `LIBUCKS_EVAL_REPOS=echoswarm` smoke test runs without error |
| **Baseline eval** on echoswarm with current Phase 4-A architecture (4 paths) | echoswarm baseline numbers saved | Verifies the existing system works on the new repo; gives `hybrid` baseline for the A/B |

**Gate before moving to 4-C.2**: echoswarm hybrid grounding ≥ 12/25
(48%, proportional to libugry's 19.5/30 = 65%). If much lower, the
fixture set is too hard or buckets are off — pause and reconsider.

### 4-C.1.5 — Query-aware Librarian (~1-2 days) — INSERTED 2026-06-15

**Why this exists:** Phase 4-C.1's echoswarm baseline exposed a real
architectural blind spot: `Librarian._handle_query` reads the bucket
via positional `_collect_source_text(max_chars=3000)`. On libugry
(small files, 50-100 LOC) buckets typically have 1-3 chunks → positional
read captures most content. On echoswarm (files 400-700 LOC) buckets
have 10-23 chunks → Librarian sees only 3-30% of bucket content. This
is the reason `latent` on echoswarm is 2/25 (vs 5/30 on libugry).

The Phase 3-B chunk-rerank fix touched only the verbatim channel
(`Translator._gather_verbatim`); the Librarian's encoder input was
left positional. Two consumers of `_collect_source_text` exist; we
only patched one. This sub-phase patches the other.

**Architectural argument** (vs the alternative of resizing buckets):
the buckets themselves are at the right granularity (one bucket per
file or per major module). Per-repo bucket sizing breaks the
universal-architecture claim. The right fix is to make the *consumers*
read content query-aware, exactly as we did for verbatim.

| step | output |
|---|---|
| Inject `ChunkRetriever` into `Librarian` (constructor arg, optional) | new field |
| Rewrite `Librarian._handle_query` to use `chunk_retriever.score_chunks(bucket_id, q_emb)` + greedy budget fill, falling back to positional when retriever unavailable | new code path |
| Compute `q_emb = self._embedder.embed(event.query)` in Librarian (embedder is already injected) | reuses Phase 2 plumbing |
| Add `LIBUCKS_LIBRARIAN_QUERY_AWARE=1` env gate in eval harness + mcp_bridge + cli — default OFF for backward compat | matches LIBUCKS_DISABLE_ROUTING_RERANK pattern |
| **Smoke test (inference only, no retraining)**: run echoswarm eval with gate ON | measures lift without distribution-shift handling |
| If smoke shows lift ≥+2 grounding: retrain echoswarm adapter+LoRA with query-aware Librarian inputs at training time (consistent distribution) | ~2-3h training |
| If smoke shows NO lift: input isn't the bottleneck, soft-prompt capacity is — skip retrain, document, proceed to 4-C.2 | useful finding either way |
| Final re-eval on echoswarm + libugry cross-validation | new latest.json |

**Gate before 4-C.2**:
- echoswarm latent grounding ≥ 5/25 (from 2/25) → architecture-input
  blind spot fixed; cache aug A/B in 4-C.2 becomes fair.
- OR cleanly documented "no lift; soft prompt is bottleneck" finding —
  cache aug A/B then directly tests the soft-prompt-capacity hypothesis.
- libugry hybrid grounding within ±2 of 19.5 on cross-validation → no
  regression on the smaller-file repo.

### 4-C.2 — KV cache extraction + storage (~2 days)

Build the per-bucket cache pipeline.

| step | output |
|---|---|
| `libucks/cache_augmentation/__init__.py` package | importable |
| `libucks/cache_augmentation/kv_extract.py`: function that runs frozen Qwen-3B forward over `bucket_source_text`, captures `past_key_values` tuple (36 layers of (K, V) tensors), returns serializable structure | Roundtrip test: extract → save → load → augmented forward gives same logits as direct forward (within float tolerance) |
| `libucks/cache_augmentation/bucket_kv_cache.py`: `BucketKVCache` class with `save(bucket_id, kv)`, `load(bucket_id)`, `invalidate(bucket_id)` | Per-bucket file under `.libucks/kv_cache/<bucket_id>.safetensors` |
| Storage format: safetensors, bf16 (start), int8 quantization (later optimization) | ~18 MB/bucket, ~540 MB for 30 buckets |
| Hook into mitosis/merge events: when a bucket changes, invalidate its cache | Test: trigger mitosis, assert cache regenerates |
| Hook into indexing: when a new bucket is created, build its cache | Test: spawn novel bucket, assert cache file exists |

**Gate before 4-C.3**: KV roundtrip is correct, caches build for all
echoswarm buckets in <2 min total.

### 4-C.3 — Coprocessor architecture (~3 days)

Implement the lightweight coprocessor as designed.

| step | output |
|---|---|
| `libucks/cache_augmentation/coprocessor.py`: `Coprocessor` class | nn.Module with 8 transformer blocks + cross-attention to bucket KV |
| Soft token embeddings: `(K=64, 2048)` learnable, init from N(0, 0.02) | Trainable parameter |
| Cross-attention layers: queries from soft tokens, keys+values from bucket KV cache (projected from per-layer KV → fused representation) | Cross-attn module |
| Weight init from middle layers of Qwen-3B (layers 16-23) | Loads from base weights |
| Forward: `coproc(bucket_kv, soft_tokens) → z ∈ (K=64, 2048)` | Output dim 2048 (matches receiver) |
| Smoke test: random bucket KV + soft tokens → z; output shape and dtype correct | Pass |

**Gate before 4-C.4**: coprocessor forward runs without OOM on MPS,
output stats (norm, mean, std) reasonable (within 2× of frozen
receiver's intermediate activations).

### 4-C.4 — Cross-bucket fusion + cache-aware decode (~2 days)

Wire the rest of the pipeline.

| step | output |
|---|---|
| `libucks/cache_augmentation/fusion.py`: `CrossBucketFusion` class | Repurpose `CommunicationAdapter` structure |
| Warm-start fusion weights from existing `adapter.pt` | `load_saved_weights` |
| Forward: `fusion([z_1, z_2, z_3]) → z_fused` | Output (K=64, 2048) |
| `libucks/cache_augmentation/decode.py`: function that takes z_fused + verbatim + query → augmented KV cache → generate | Uses `model.generate(past_key_values=C_aug, ...)` |
| Build augmented cache: run frozen receiver forward over z_fused → C_z, run forward over verbatim+query → C_input, concatenate layer-by-layer | torch tensor manipulation |
| Add `cache_aug` mode to `Translator`: new `synthesize_cache_aug()` method calling the above | Preserves "Translator is only decode" constraint |
| Wire into eval harness as path `cache_aug` | Path appears in eval JSON |

**Gate before 4-C.5**: end-to-end inference produces coherent text
(even if untrained), no crashes, no NaNs.

### 4-C.5 — Training data + multi-position augmentation training (~5 days)

The hardest piece. Most likely to need iteration.

| step | output |
|---|---|
| Reuse `libucks/thinking/training/data_generator.py` to synthesize Q&A pairs from echoswarm buckets | ~5K training pairs |
| New trainer: `libucks/thinking/training/cache_aug_trainer.py` | Replaces `_train_lora_receiver` |
| Multi-position augmentation: for each training example, insert latent embeddings at N_T positions and predict N_A=16 ahead tokens (DeepMind §2.3). Construct custom attention mask | Custom forward pass |
| Token Assorted curriculum: per-example sample `text_ratio` uniformly | Randomized text/latent mix |
| Loss: cross-entropy on ahead tokens | Standard LM loss |
| Optimizer: AdamW, lr 1e-4, warmup 500 steps, total ~3 epochs over training set | Hyperparams |
| Training run on MPS, ~2-3 days wall | `cache_aug_state.pt` saved to repo's `.libucks/` |
| Convergence check: training loss decreases, validation perplexity drops | Match DeepMind's qualitative pattern |

**Gate before 4-C.6**: validation perplexity at position 1 is lower
than baseline (without coprocessor) by ≥3%. Otherwise something's
wrong — investigate before evaluating.

**Fallback if training is unstable**: switch to Compressed CoT teacher-
forcing recipe (gold hidden states + scaled MSE, layer-by-layer). +2 days.

**Fallback if quality is insufficient at this scale**: escalate to full
Qwen-3B coprocessor LoRA-128 per DeepMind. +5 days, +6 GB MPS memory
(may not fit; would need gradient checkpointing + lower batch).

### 4-C.6 — Eval on echoswarm + libugry cross-validation (~1 day)

| step | output |
|---|---|
| Run 5-path eval on echoswarm: latent, hybrid, **cache_aug**, text_clean, no_context | `tests/eval/results/phase4c/echoswarm_<timestamp>.json` |
| Per-question diff: where does cache_aug win/lose vs hybrid? | Diagnostic dump in stderr |
| Cross-validation: rerun the trained coprocessor on libugry (without retraining) | Confirms not echoswarm-specific |
| 2-run minimum for noise estimate on the cache_aug headline | Mean ± std on the new path |

**Gate**: at this point, the data picks Phase 4-D.

### 4-C.7 — Decision + writeup of result (~half day)

Based on the table:

- **4-D-A migrate**: archive `adapter.pt` / `lora_receiver.pt` / `w_a.pt`
  as Phase-4-A baseline weights. Make `cache_aug` the default `cache_aug_retrieval = true` config. Update CLAUDE.md golden rules.
- **4-D-B dual-track**: keep both modes. `libucks_query(mode=)` parameter.
- **4-D-C future-work**: roll back to Phase 4-A. Cache aug becomes
  related-work in the writeup.

Save `project_phase_4c_result.md` recording numbers + decision.

## Critical files

### NEW (Phase 4-C)
- `libucks/cache_augmentation/__init__.py`
- `libucks/cache_augmentation/kv_extract.py` — KV cache from frozen
  receiver forward over bucket text
- `libucks/cache_augmentation/bucket_kv_cache.py` — load/save/invalidate
- `libucks/cache_augmentation/coprocessor.py` — lightweight coprocessor
- `libucks/cache_augmentation/fusion.py` — cross-bucket fusion (warm-start
  from current adapter)
- `libucks/cache_augmentation/decode.py` — augmented decode (Translator
  callable)
- `libucks/thinking/training/cache_aug_trainer.py` — multi-position
  training loop with Token Assorted curriculum
- `tests/integration/test_cache_aug_smoke.py` — KV roundtrip + invalidation
  tests
- `tests/eval/fixtures/echoswarm_qa.json` — ~25 hand-curated fixtures

### MODIFIED
- `libucks/translator.py` — add `synthesize_cache_aug()` method
- `libucks/storage/bucket_store.py` — frontmatter extended with
  `kv_cache_path` (Optional[str])
- `libucks/storage/bucket_registry.py` — track per-bucket cache freshness
- `libucks/config.py` — new `[cache_augmentation]` section
- `libucks/_cli.py` — `libucks train-cache-aug` command
- `libucks/mcp_bridge.py` — wire cache aug into the query handler
- `libucks/health_monitor.py` — invalidate cache on mitosis/merge
- `tests/eval/test_latent_vs_baseline.py` — add `cache_aug` path; extend
  `_REPOS` with echoswarm

### PRESERVED (no changes)
- `libucks/central_agent.py`
- `libucks/chunk_retriever.py`
- `libucks/embeddings/embedding_service.py`
- `libucks/novel_bucket_service.py`, `mitosis.py`, `merging_service.py`
- `libucks/git_hook_receiver.py`
- `libucks/librarian.py` (Librarian still emits Representation for the
  `latent` and `hybrid` eval paths; cache aug is additive)

## Verification end-to-end

### After 4-C.2 (KV cache layer)
1. Roundtrip: extract bucket KV → save → load → augmented forward
   logits identical (within 1e-3) to direct forward over same text.
2. Cache invalidation: trigger mitosis on an echoswarm bucket; assert
   parent cache deleted, child caches built.
3. `uv run pytest tests/integration/test_cache_aug_smoke.py` passes.

### After 4-C.4 (decode path)
1. End-to-end inference: random echoswarm query → cache_aug path runs
   without crash; output is coherent (even if untrained quality is low).
2. Memory ceiling: peak MPS allocation < 14 GB during inference.

### After 4-C.5 (trained model)
1. Training loss curve: monotonically decreasing.
2. Validation perplexity on echoswarm holdout: position-1 perp <
   baseline-1 perp by ≥3%.
3. `cache_aug_state.pt` < 1 GB on disk.

### After 4-C.6 (eval)
1. `tests/eval/results/phase4c/echoswarm_*.json` has 5 paths populated.
2. Cross-validation libugry run: cache_aug numbers in same ballpark
   (within ±3 grounding).
3. Per-question diff identifies wins and losses.
4. Memory entry `project_phase_4c_result.md` saved.

## Time + compute budget

**Total**: ~15 working days ≈ **3 weeks elapsed wall-time** (accounting
for compute that runs while we sleep).

| sub-phase | working days | MPS compute (wall) |
|---|---|---|
| 4-C.1 infra | 2 | 1 eval (~1.5h) |
| 4-C.2 KV cache | 2 | smoke tests |
| 4-C.3 coprocessor | 3 | smoke tests |
| 4-C.4 decode | 2 | smoke tests |
| 4-C.5 training | 5 | **2-3 days continuous training** |
| 4-C.6 eval | 1 | 2 evals (~3h) |
| 4-C.7 decision | 0.5 | — |

Cost: $0 monetary. Local MPS compute, no API calls.

## Risks and mitigations

| risk | likelihood | mitigation |
|---|---|---|
| MPS OOM during training (3B frozen + 250M coprocessor + gradients) | medium | Gradient checkpointing on the coprocessor, batch size 1, accumulation; lower coprocessor to 4 blocks if needed |
| Multi-position attention mask construction is buggy | high | Smoke tests on a 4-position toy sequence before scaling. Validate against simple "single-position" reference loss. |
| KV cache I/O is slow (per-bucket disk reads) | low | Memory cache in addition to disk cache (current ChunkRetriever pattern). |
| Coprocessor outputs are too noisy / not informative | medium | Compressed CoT teacher-forcing fallback (4-C.5 extension). |
| Echoswarm fixture quality is uneven | medium | Self-generate from source code (libugry pattern); manual filter to ≥20 keepers. |
| Cross-bucket fusion architecture choice is wrong | medium | Try concatenate-then-coprocess (option A) as ablation if hierarchical (option C) underperforms. |
| Training data too sparse (echoswarm small repo) | medium | Augment with libugry pairs (cross-repo transfer test built into 4-C.6). |
| `cache_aug` < `hybrid` after all this work | possible | 4-D-C is a defined outcome, not a failure — paper still publishable with cache aug as related work. |

## What this plan does NOT do

- Does NOT touch `libugry`'s buckets or training weights.
- Does NOT change the routing layer (centroid only, Phase 4-A decision).
- Does NOT modify click weights.
- Does NOT change the chunk_retriever for verbatim selection.
- Does NOT build the Heima interpreter (Phase 5+).
- Does NOT do per-query coprocessor calls (would multiply inference
  cost) — the coprocessor runs once at indexing time per bucket;
  fusion runs per query.
- Does NOT depend on remote compute or API access.

## Open questions deferred to during-execution

1. Coprocessor block count: 8 is the default; if too slow or
   memory-heavy, drop to 6 or 4. Defer to 4-C.3.
2. Coprocessor weight initialization: middle layers (16-23) vs
   early (0-7) vs late (28-35). Defer to 4-C.3 small ablation.
3. Multi-position N_T: DeepMind uses 128, may be too dense for our
   ~500-token training pairs. Try 16, 32, 64. Defer to 4-C.5.
4. Verbatim channel in `cache_aug` mode: full Phase 4-A verbatim
   (1200 chars/bucket via chunk rerank) vs reduced/no verbatim
   to isolate cache aug contribution. Run BOTH in 4-C.6 as
   ablations (`cache_aug` with verbatim and `cache_aug_no_verbatim`).
