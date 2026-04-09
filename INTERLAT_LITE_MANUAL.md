# Interlat-Lite: End-to-End Instruction Manual

**Phase 12 complete.** This document is the ground-truth runbook for running the new pipeline on a real repository.

---

## 0. What Phase 12 Built (and What It Didn't)

Phase 12 delivered seven building blocks, all tested:

| Built | File | Purpose |
|---|---|---|
| `CurriculumMixer` | `libucks/thinking/curriculum.py` | Blends token embeds + latents per training step |
| `CommunicationAdapter.frame()` + 8-head MHA | `communication_adapter.py` | Wraps soft-prompt with `<bop>`/`<eop>` boundaries |
| `ModelManager.load_base_model()` | `model_manager.py` | Loads a separate Base model for the receiver |
| `separation_loss()` | `training/losses.py` | JSD-based L_sep so model can't ignore latents |
| `generate_curriculum_batch()` | `training/data_generator.py` | Produces mixed-input + target-id training items |
| `LoRAReceiverTrainer` | `training/lora_trainer.py` | Fine-tunes q_proj/v_proj of Base model |
| `LatentStrategy.receive()` | `latent_strategy.py` | New decode path — no NormMatch, no Residual Anchoring |

**What is NOT yet wired (as of Phase 12):**

- The `libucks train-adapter` CLI still uses `ContrastiveAdapterTrainer` (InfoNCE/MSE — the old approach).
- The `Translator._synthesize_latent()` still calls `strategy.decode()`, not `strategy.receive()`.
- `ModelConfig` has no `base_model` field yet.

This means you cannot yet run `libucks query` and get Interlat-Lite output. You run the two glue scripts in Sections 3 and 4 instead. Wiring these into the CLI is Phase 13.

---

## 1. Prerequisites

### Hardware

- **Apple Silicon (M1/M2/M3) with ≥ 16 GB unified memory.** The Interlat-Lite pipeline loads **two** models simultaneously:
  - `Qwen2.5-3B-Instruct` (~6 GB float16) — encoder / Librarian reasoning
  - `Qwen2.5-3B-Base` + LoRA (~6 GB float16) — receiver / Translator decoding
  - Total resident: ~12 GB. 8 GB machines will swap and be unusably slow.
- **CUDA GPU with ≥ 16 GB VRAM** also works. Set `device = "cuda"` everywhere below.
- CPU-only fallback exists but training will take hours.

### Python environment

```bash
# From the libucks repo root
pip install -e ".[latent]"
pip install peft   # Phase 12 LoRA dependency — not yet in pyproject.toml extras
```

Verify:

```bash
python -c "import torch, peft, transformers; print(torch.__version__, peft.__version__, transformers.__version__)"
```

### Anthropic API key

Still required during training (V1 teacher generates the curriculum text targets):

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Two model IDs

| Role | Model ID | Notes |
|---|---|---|
| Encoder (Librarians) | `Qwen/Qwen2.5-3B-Instruct` | Already used in V2.0 |
| Receiver (Translator) | `Qwen/Qwen2.5-3B` | Base model — ~6 GB download on first run |

Both are pulled from HuggingFace automatically on first use.

### Existing buckets

You must have already run `libucks init` on your target repo. Training uses the existing bucket prose as the dataset.

```bash
libucks init --local /path/to/libugry
```

---

## 2. Understand the New Pipeline (5-Minute Concept)

**Before Phase 12 (broken):**
```
Librarians → hidden states → CommunicationAdapter → soft-prompt
→ NormMatch + Residual Anchoring → frozen Instruct model → character soup
```

**After Phase 12 (correct):**
```
Librarians   → hidden states (from Instruct encoder)
                    ↓
             CommunicationAdapter
                    ↓
             adapter.frame(soft_prompt, bop_embed, eop_embed)  ← <bop> h1..hK <eop>
                    ↓
             LoRA-trained Base receiver (via strategy.receive())
                    ↓
             coherent English
```

**Key insight:** The Base receiver was trained by seeing `[token_embeds | latents]` mixtures via curriculum sampling. It learned to read latents by starting from the known token-embedding manifold and gradually shifting to pure latents. No post-hoc manifold correction needed.

**LoRA training is a one-time step per encoder model** (not per repo). Once `base_receiver_lora.pt` exists for `Qwen2.5-3B-Instruct` → `Qwen2.5-3B`, you can re-use it across repos. The adapter weights (`adapter.pt`) remain repo-specific.

---

## 3. Training Step — The LoRA Receiver

### 3a. Run the training glue script

Save this as `scripts/train_lora_receiver.py` in your libucks repo:

```python
"""Train the Interlat-Lite LoRA receiver.

Run from the libucks repo root:
    python scripts/train_lora_receiver.py --repo /path/to/libugry --epochs 3
"""
import asyncio
import argparse
from pathlib import Path


async def main(repo_path: Path, epochs: int, device: str, output: Path) -> None:
    import torch
    from libucks.config import Config
    from libucks.thinking.model_manager import ModelManager
    from libucks.thinking.latent_strategy import LatentStrategy
    from libucks.thinking.communication_adapter import CommunicationAdapter
    from libucks.thinking.training.data_generator import MultiPerspectiveDataGenerator
    from libucks.thinking.training.lora_trainer import LoRAReceiverTrainer
    from libucks.thinking.text_strategy import TextStrategy
    from libucks.storage.bucket_store import BucketStore
    from libucks.storage.bucket_registry import BucketRegistry

    cfg = Config.load(repo_path)
    bucket_dir = repo_path / ".libucks"

    # ── Load models ──────────────────────────────────────────────────────────
    print(f"[train] Loading Instruct encoder on {device}...")
    mgr = ModelManager()
    mgr.load("Qwen/Qwen2.5-3B-Instruct", quantization="none", device=device)

    print(f"[train] Loading Base receiver on {device}...")
    mgr.load_base_model("Qwen/Qwen2.5-3B", quantization="none", device=device)

    strategy = LatentStrategy(mgr)

    # ── Load buckets ──────────────────────────────────────────────────────────
    registry = BucketRegistry(repo_path / cfg.paths.registry_file)
    registry.load()
    store = BucketStore(repo_path / cfg.paths.bucket_dir)
    bucket_ids = list(registry.get_all_centroids().keys())
    if not bucket_ids:
        print("[train] ERROR: No buckets found. Run `libucks init` first.")
        return
    print(f"[train] {len(bucket_ids)} buckets found")

    # ── Build adapter and training components ─────────────────────────────────
    adapter = CommunicationAdapter()
    adapter.load_saved_weights(bucket_dir / "adapter.pt")
    if device == "mps":
        adapter = adapter.to(device=device, dtype=torch.float16)
    else:
        adapter = adapter.to(device=device)

    text_strategy = TextStrategy.from_env(cfg.model.anthropic_model)
    generator = MultiPerspectiveDataGenerator(
        text_strategy=text_strategy,
        latent_strategy=strategy,
        registry=registry,
        store=store,
    )

    base_model = mgr.get_base_model()
    base_tokenizer = mgr.get_base_tokenizer()
    embedding_layer = base_model.model.embed_tokens

    # Register <bop>/<eop> special tokens if not already present
    special_tokens = {"additional_special_tokens": ["<bop>", "<eop>"]}
    n_added = base_tokenizer.add_special_tokens(special_tokens)
    if n_added > 0:
        print(f"[train] Added {n_added} special token(s) (<bop>, <eop>) to base tokenizer")
        base_model.resize_token_embeddings(len(base_tokenizer))

    trainer = LoRAReceiverTrainer(base_model, lora_r=16, lora_alpha=32, lr=1e-4)

    # ── Training loop ──────────────────────────────────────────────────────────
    bop_id = base_tokenizer.convert_tokens_to_ids("<bop>")
    eop_id = base_tokenizer.convert_tokens_to_ids("<eop>")
    bop_embed = embedding_layer(
        torch.tensor([bop_id], device=device)
    ).squeeze(0).detach()
    eop_embed = embedding_layer(
        torch.tensor([eop_id], device=device)
    ).squeeze(0).detach()

    print(f"[train] Starting {epochs} epoch(s) of LoRA training...")
    for epoch in range(epochs):
        epoch_task_loss = 0.0
        epoch_sep_loss = 0.0
        count = 0
        for i, bucket_id in enumerate(bucket_ids, 1):
            print(f"  Epoch {epoch+1}/{epochs}  bucket {i}/{len(bucket_ids)}: {bucket_id}")
            try:
                item = await generator.generate_curriculum_batch(
                    bucket_id=bucket_id,
                    adapter=adapter,
                    tokenizer=base_tokenizer,
                    embedding=embedding_layer,
                    output_len=adapter.output_len,
                    hidden_dim=adapter.hidden_dim,
                )
                # Add wrong_latent: encode a different bucket's prose as the mismatch
                wrong_id = bucket_ids[(i % len(bucket_ids))]  # adjacent bucket
                _, wrong_prose = store.read(wrong_id)
                wrong_latent = await strategy.encode(wrong_prose)
                wrong_soft = adapter([wrong_latent.to(device)])
                framed_wrong = adapter.frame(wrong_soft, bop_embed, eop_embed)
                item["wrong_latent"] = framed_wrong[1:-1]  # strip bop/eop for L_sep

                losses = trainer.train_step(item)
                epoch_task_loss += losses["task"]
                epoch_sep_loss += losses["sep"]
                count += 1
                print(f"    L_task={losses['task']:.4f}  L_sep={losses['sep']:.4f}")
            except Exception as exc:
                print(f"  Skipped {bucket_id}: {exc}")

        if count:
            print(f"  Epoch {epoch+1} avg — L_task={epoch_task_loss/count:.4f}  "
                  f"L_sep={epoch_sep_loss/count:.4f}")

    # ── Save LoRA weights ──────────────────────────────────────────────────────
    import torch
    lora_params = {n: p for n, p in base_model.named_parameters() if "lora" in n.lower()}
    torch.save(lora_params, output)
    print(f"[train] LoRA weights saved to {output}  ({len(lora_params)} tensors)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--output", default=None, type=Path,
                        help="Output path for LoRA weights (default: <repo>/.libucks/base_receiver_lora.pt)")
    args = parser.parse_args()

    output = args.output or (Path(args.repo) / ".libucks" / "base_receiver_lora.pt")
    asyncio.run(main(Path(args.repo), args.epochs, args.device, output))
```

### 3b. Run it on libugry

```bash
cd /path/to/libucks
python scripts/train_lora_receiver.py \
    --repo /path/to/libugry \
    --epochs 3 \
    --device mps
```

**Expected output:**

```
[train] Loading Instruct encoder on mps...
[train] Loading Base receiver on mps...
[train] 8 buckets found
[train] Added 2 special token(s) (<bop>, <eop>) to base tokenizer
[train] Starting 3 epoch(s) of LoRA training...
  Epoch 1/3  bucket 1/8: bucket-auth
    L_task=4.1832  L_sep=0.0041
  Epoch 1/3  bucket 2/8: bucket-db
    L_task=3.9204  L_sep=0.0088
  ...
  Epoch 1 avg — L_task=4.0123  L_sep=0.0063
  ...
  Epoch 3 avg — L_task=2.1847  L_sep=0.0291
[train] LoRA weights saved to /path/to/libugry/.libucks/base_receiver_lora.pt  (72 tensors)
```

**What healthy training looks like:**

| Signal | What to watch |
|---|---|
| `L_task` | Should decrease across epochs: 4.x → 2.x → 1.x is good. Staying above 3.5 after epoch 2 means lr is too low — try `--lr 5e-4` |
| `L_sep` | Should slowly increase as training progresses (model learns to differentiate correct vs wrong latents). A flat `L_sep ≈ 0.000` means the model is ignoring latents — check `wrong_latent` construction |
| Wall time | On MPS with Qwen2.5-3B: ~3–5 minutes per bucket per epoch. For 8 buckets × 3 epochs ≈ 1–2 hours. First run will be longer (model download) |

**If L_task never decreases:** The learning rate is too high (exploding gradients) or too low. Try `--lr 3e-4`. Also verify that `adapter.pt` exists — if the adapter is random, the latents it produces are also random and cross-entropy loss is harder to fit.

**Do I need to retrain for every repo?** No. The LoRA weights encode how to interpret Instruct-generated latents in general. Once trained on any repo with reasonable coverage (≥ 5 buckets), the weights transfer. You only need to retrain if you upgrade the encoder model (e.g., switch from 3B to 7B Instruct).

---

## 4. Inference Step — Running a Query

### 4a. The inference glue script

Save this as `scripts/query_interlat.py`:

```python
"""Run an Interlat-Lite query — full pipeline with receive() path.

Usage:
    python scripts/query_interlat.py \
        --repo /path/to/libugry \
        --query "What is the main purpose of this code?" \
        --lora /path/to/libugry/.libucks/base_receiver_lora.pt
"""
import asyncio
import argparse
import sys
from pathlib import Path


async def main(repo_path: Path, query_text: str, lora_path: Path, top_k: int, device: str):
    import torch
    from libucks.config import Config
    from libucks.thinking.model_manager import ModelManager
    from libucks.thinking.latent_strategy import LatentStrategy
    from libucks.thinking.communication_adapter import CommunicationAdapter
    from libucks.embeddings.embedding_service import EmbeddingService
    from libucks.storage.bucket_registry import BucketRegistry
    from libucks.storage.bucket_store import BucketStore
    from libucks.central_agent import CentralAgent
    from libucks.librarian import Librarian
    from libucks.query_orchestrator import QueryOrchestrator

    cfg = Config.load(repo_path)
    bucket_dir = repo_path / ".libucks"

    # ── Load both models ──────────────────────────────────────────────────────
    print(f"[query] Loading Instruct encoder ({device})...", file=sys.stderr)
    mgr = ModelManager()
    mgr.load("Qwen/Qwen2.5-3B-Instruct", quantization="none", device=device)

    print(f"[query] Loading Base receiver ({device})...", file=sys.stderr)
    mgr.load_base_model("Qwen/Qwen2.5-3B", quantization="none", device=device)

    # ── Restore LoRA weights ──────────────────────────────────────────────────
    if lora_path.exists():
        print(f"[query] Loading LoRA weights from {lora_path}...", file=sys.stderr)
        base_model = mgr.get_base_model()
        lora_state = torch.load(lora_path, map_location=device, weights_only=True)
        # Load only the LoRA parameter tensors into the model
        current = dict(base_model.named_parameters())
        loaded = 0
        for name, tensor in lora_state.items():
            if name in current:
                current[name].data.copy_(tensor)
                loaded += 1
        print(f"[query] Restored {loaded} LoRA tensors", file=sys.stderr)
    else:
        print(f"[query] WARNING: {lora_path} not found — using untrained receiver", file=sys.stderr)

    strategy = LatentStrategy(mgr)

    # ── Register <bop>/<eop> tokens ───────────────────────────────────────────
    base_tokenizer = mgr.get_base_tokenizer()
    base_model = mgr.get_base_model()
    base_tokenizer.add_special_tokens({"additional_special_tokens": ["<bop>", "<eop>"]})
    base_model.resize_token_embeddings(len(base_tokenizer))

    bop_id = base_tokenizer.convert_tokens_to_ids("<bop>")
    eop_id = base_tokenizer.convert_tokens_to_ids("<eop>")
    embedding_layer = base_model.model.embed_tokens
    bop_embed = embedding_layer(torch.tensor([bop_id], device=device)).squeeze(0).detach()
    eop_embed = embedding_layer(torch.tensor([eop_id], device=device)).squeeze(0).detach()

    # ── Load buckets + routing ────────────────────────────────────────────────
    registry = BucketRegistry(repo_path / cfg.paths.registry_file)
    registry.load()
    store = BucketStore(repo_path / cfg.paths.bucket_dir)
    bucket_ids = list(registry.get_all_centroids().keys())
    if not bucket_ids:
        print("[query] ERROR: No buckets. Run `libucks init` first.", file=sys.stderr)
        return

    _real_stdout = sys.stdout; sys.stdout = sys.stderr
    try:
        embedder = EmbeddingService.get_instance(cfg.model.embedding_model)
    finally:
        sys.stdout = _real_stdout

    # ── Load adapter ──────────────────────────────────────────────────────────
    adapter = CommunicationAdapter()
    adapter.load_saved_weights(bucket_dir / "adapter.pt")
    adapter_dtype = torch.float16 if device == "mps" else None
    adapter = adapter.to(device=device, dtype=adapter_dtype)

    # ── Build librarians + orchestrator ───────────────────────────────────────
    agent = CentralAgent(registry, cfg, embed_fn=embedder.embed)
    librarians = {}
    for bid in bucket_ids:
        lib = Librarian(
            bucket_id=bid, store=store, registry=registry,
            strategy=strategy, embedder=embedder,
            mitosis_threshold=cfg.routing.mitosis_threshold,
        )
        librarians[bid] = lib
        agent.register_librarian(bid, lib)

    orchestrator = QueryOrchestrator(
        central_agent=agent, librarians=librarians,
        embed_fn=embedder.embed, top_k=top_k,
    )

    # ── Query → representations → frame → receive ─────────────────────────────
    print(f'[query] Routing: "{query_text}"', file=sys.stderr)
    representations = await orchestrator.query(query_text)
    print(f"[query] {len(representations)} representations from Librarians", file=sys.stderr)

    if not representations:
        print("No relevant context found.", file=sys.stderr)
        return

    # Merge via adapter, frame with <bop>/<eop>, decode via Base receiver
    contiguous_reps = [r.contiguous() for r in representations]
    with torch.no_grad():
        soft_prompt = adapter(contiguous_reps)            # (K, D)
    framed = adapter.frame(soft_prompt, bop_embed, eop_embed)   # (K+2, D)

    print("[query] Calling strategy.receive()...", file=sys.stderr)
    answer = await strategy.receive(framed)

    # Answer to stdout; all logs to stderr
    print(answer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--query", required=True)
    parser.add_argument("--lora", default=None, type=Path)
    parser.add_argument("--top-k", default=3, type=int)
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    args = parser.parse_args()

    lora = args.lora or (Path(args.repo) / ".libucks" / "base_receiver_lora.pt")
    asyncio.run(main(Path(args.repo), args.query, lora, args.top_k, args.device))
```

### 4b. Run a query against libugry

```bash
python scripts/query_interlat.py \
    --repo /path/to/libugry \
    --query "What is the main purpose of this code?" \
    --device mps
```

Or for the fastest single-bucket smoke test:

```bash
python scripts/query_interlat.py \
    --repo /path/to/libugry \
    --query "What does this module do?" \
    --top-k 1 \
    --device mps
```

**Expected successful output (stderr + stdout):**

```
[query] Loading Instruct encoder (mps)...
[query] Loading Base receiver (mps)...
[query] Loading LoRA weights from ...base_receiver_lora.pt...
[query] Restored 72 LoRA tensors
[query] Routing: "What is the main purpose of this code?"
[libucks:latent] reason: tokenizing (1842 chars, device=mps)
[libucks:latent] reason: forward pass (seq_len=256)
...
[query] 3 representations from Librarians
[query] Calling strategy.receive()...
[libucks:latent] receive: framed_latent (34, 2048), device=mps
[libucks:latent] receive: generated 87 tokens
[libucks:latent] receive: decode complete (312 chars)

This module implements a lightweight authentication layer that validates
incoming requests using HMAC signatures. It provides three main functions:
verify_signature(), which checks the request header against a shared secret...
```

---

## 5. Testing Methodology: Verifying the Character-Soup is Gone

### Test 1 — Coherence smoke test (2 minutes)

Run the query without LoRA (untrained receiver) to establish the broken baseline, then with LoRA:

```bash
# Baseline: untrained receiver (will produce character-soup or random output)
python scripts/query_interlat.py \
    --repo /path/to/libugry \
    --query "What does the auth module do?" \
    --lora /nonexistent/path.pt \
    --top-k 1 2>/dev/null

# After training: trained LoRA receiver
python scripts/query_interlat.py \
    --repo /path/to/libugry \
    --query "What does the auth module do?" \
    --top-k 1 2>/dev/null
```

**Pass criteria:** The trained output contains alphabetic English words, complete sentences, and at least one term that appears in the actual source code of libugry. The untrained output should be visibly incoherent (\\., $s, BPE fragments) — if it looks coherent without LoRA, the manifold collapse hasn't been triggered yet.

### Test 2 — Factual grounding check (5 minutes)

Ask a question with a specific, verifiable answer:

```bash
python scripts/query_interlat.py \
    --repo /path/to/libugry \
    --query "What function handles the main entry point?" \
    --top-k 2 2>/dev/null
```

Open the actual libugry source and verify that the function name in the answer exists in the codebase. This confirms the latent communication is carrying semantic content, not just producing fluent hallucinations.

### Test 3 — L_task loss convergence check

Inspect training logs after epoch 1 and epoch 3:

```
Epoch 1 avg — L_task=4.01   ← random-ish, ~log(vocab)
Epoch 3 avg — L_task=1.85   ← model fitting the curriculum
```

If `L_task` after epoch 3 is still above 3.5, the receiver hasn't learned. Diagnose:

1. **Is `adapter.pt` valid?** Run `libucks train-adapter --creative --epochs 3` first to ensure the adapter has meaningful weights before training the receiver.
2. **Is the learning rate correct?** The default `lr=1e-4` is conservative. For faster convergence on small bucket sets (< 10 buckets), try `1e-3`.
3. **Is `peft` actually modifying weights?** Add this check to the script after `LoRAReceiverTrainer()`:
   ```python
   lora_params = [(n, p) for n, p in trainer.model.named_parameters() if "lora" in n]
   print(f"LoRA params: {len(lora_params)}, total trainable: {sum(p.numel() for _, p in lora_params)}")
   ```
   You should see ~144 LoRA params (2 per attention layer × 36 layers × 2 projections).

### Test 4 — Cross-query consistency

Ask the same semantic question phrased two different ways:

```bash
python scripts/query_interlat.py --repo /path/to/libugry \
    --query "How does the system handle errors?" --top-k 2 2>/dev/null

python scripts/query_interlat.py --repo /path/to/libugry \
    --query "What is the error handling strategy?" --top-k 2 2>/dev/null
```

Both answers should reference the same concepts and the same code locations. Inconsistent answers (one mentions try/catch, the other mentions logging) suggest the adapter hasn't been trained well — run more adapter training epochs first (`libucks train-adapter --creative --epochs 5`).

---

## 6. The Correct Order of Operations

```
Step 0: libucks init --local /path/to/libugry
           ↓  (creates .libucks/ with bucket prose)

Step 1: libucks train-adapter --creative --epochs 3
           ↓  (trains CommunicationAdapter → .libucks/adapter.pt)
           ↓  (still uses old InfoNCE loss — this is correct, adapter training is separate)

Step 2: python scripts/train_lora_receiver.py --repo /path/to/libugry --epochs 3
           ↓  (trains LoRA on Base receiver → .libucks/base_receiver_lora.pt)
           ↓  (uses adapter.pt from Step 1 to generate soft-prompts)

Step 3: python scripts/query_interlat.py --repo /path/to/libugry --query "..."
           ↓  (full Interlat-Lite inference)
```

Steps 1 and 2 are **independent** — you can retrain either without re-running the other. Step 1 must be completed before Step 2 (the LoRA trainer needs a meaningful adapter to generate sensible soft-prompts).

---

## 7. Artifacts Produced

| File | Created by | Contains | Re-run when? |
|---|---|---|---|
| `.libucks/buckets/` | `libucks init` | Bucket prose markdown | New repo or major refactor |
| `.libucks/registry.json` | `libucks init` | Centroid embeddings | Same as above |
| `.libucks/adapter.pt` | `libucks train-adapter` | CommunicationAdapter weights | Buckets change significantly |
| `.libucks/base_receiver_lora.pt` | `train_lora_receiver.py` | LoRA q_proj/v_proj deltas | Encoder model changes (not per-repo) |

---

## 8. What Phase 13 Will Wire

Once Phase 13 is complete, the following CLI upgrades will make these glue scripts unnecessary:

1. `ModelConfig.base_model` field in `config.toml`
2. `libucks train-receiver` command (wraps `train_lora_receiver.py`)
3. `Translator._synthesize_latent()` calling `strategy.receive()` instead of `strategy.decode()`
4. `_run_query()` in `_cli.py` loading LoRA weights automatically

Until then, use the scripts in this document.

---

## 9. Training Strategies, Epoch Scaling & Expected Times

### 9a. Why training is slow — and what to watch for

Training has two distinct cost centers per epoch per bucket:

| Step | Cost source | Typical time (MPS, 0.5B) |
|------|------------|--------------------------|
| `text_strategy.reason()` | Anthropic API round-trip | 2–5 s |
| `strategy.reason()` / `encode()` | Local model forward pass | 3–8 s |
| `trainer.train_step()` | Backward pass (LoRA only) | 1–2 s |

**Total per bucket per epoch: ~6–15 s.** For 8 buckets × 5 epochs = 40 iterations ≈ 4–10 min total. If you're seeing 30+ min, diagnose in this order:

1. **Are train steps actually running?** Add a counter after `trainer.train_step()`. If it's 0, the broad `except` is swallowing an error — check `wrong_latent` construction and `lora_r` consistency.
2. **Is the API the bottleneck?** Add `time.time()` timestamps around `generate_curriculum_batch()`. If each call takes > 10 s, the API is the bottleneck, not the model.
3. **Is caching working?** The data generator re-calls the API each epoch. Until caching is wired in (Phase 13), pre-generate all `(bucket_id → summary_text, latent)` pairs before the epoch loop and pass them in — cuts training time by ~60%.

### 9b. Epoch scaling formula

```
adapter_epochs = max(3, ceil(30 / B))
lora_epochs    = max(2, ceil(20 / B))
```

where `B` = number of buckets. Cap at 15 / 10 respectively to avoid overfitting.

| Buckets | Adapter epochs | LoRA epochs |
|---------|---------------|-------------|
| 3–5     | 10            | 7           |
| 6–10    | 5             | 3           |
| 11–20   | 3             | 2           |
| 20+     | 3             | 2           |

**Bucket density modifier:** if `B < files/50` (few, rich buckets), add 2 epochs. If `B > files/10` (many, sparse buckets), subtract 1.

**Why fewer epochs for large repos:** each epoch already sees a diverse set of bucket contexts. The LoRA only needs to learn the latent→language mapping, not memorize bucket content — diversity within one epoch is more valuable than repetition.

### 9c. Healthy training signals

**LoRA receiver:**

| Epoch | Expected L_task | Expected L_sep |
|-------|----------------|----------------|
| 1 | 3.8–4.5 (≈ log vocab) | < 0.01 |
| 3 | 2.0–2.8 | 0.01–0.05 |
| 5+ | 1.2–2.0 | 0.05–0.15 |

`L_task` not dropping below 3.5 after epoch 2: lr too low, or `train_step` is being silently skipped — add a step counter to confirm.

`L_sep` flat at 0.000: `wrong_latent` equals `correct_latent` (wrong_id aliasing bug) or `wrong_latent` key is missing from the batch dict.

**CommunicationAdapter (InfoNCE):**

| Epoch | Expected loss |
|-------|--------------|
| 1 | 1.5–2.5 |
| 3 | 0.5–1.2 |
| 5 | 0.1–0.6 |

Loss stuck above 1.5: fewer than `min_negatives` buckets are in the similarity window — lower `NEG_SIM_LO` from 0.25 to 0.15, or use MSE fallback path for repos with < 5 buckets.

### 9d. Pre-generation cache pattern (manual workaround until Phase 13)

Until the data generator has built-in epoch-level caching, pre-generate all batch items before the epoch loop:

```python
# Before training loop:
print("[train] Pre-generating curriculum items...")
cache = {}
for bucket_id in bucket_ids:
    try:
        cache[bucket_id] = await generator.generate_curriculum_batch(
            bucket_id=bucket_id, adapter=adapter,
            tokenizer=base_tokenizer, embedding=embedding_layer,
            output_len=adapter.output_len, hidden_dim=adapter.hidden_dim,
        )
    except Exception as e:
        print(f"  [warn] Skipped {bucket_id}: {e}")

# Then in epoch loop, use cache[bucket_id] instead of calling generate_curriculum_batch
```

This converts O(epochs × buckets) API calls to O(buckets) — the single largest win available without code changes.

### 9e. LoRA rank / alpha consistency checklist

All three scripts must agree on `r` and `alpha` — a mismatch causes a hard `RuntimeError` at weight-load time:

| Script | Parameter | Must be |
|--------|-----------|---------|
| `train_lora_receiver.py` | `LoRAReceiverTrainer(lora_r=...)` | `4` |
| `query_interlat.py` | `_inject_lora(..., r=...)` | `4` |

If you change `r` for an experiment, update both. Save `r` and `alpha` into the `.pt` file header so inference can read them automatically (Phase 13 work item).
