<img width="1289" height="721" alt="Screenshot 2026-04-18 at 04 18 03" src="https://github.com/user-attachments/assets/d482ce21-c89a-4d90-816d-f09441dd3033" />

# libucks — Librarian Buckets

An experimental research project investigating alternative long context memory architectures for coding agents

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![MCP Compatible](https://img.shields.io/badge/MCP-compatible-green.svg)](https://modelcontextprotocol.io/)

---

Currently, agents interacting with large repositories suffer from context bloat, "lost in the middle" degradation, and high computational redundancy. Standard RAG pipelines attempt to solve this by repeatedly fetching and summarizing text files. However, this introduces a fundamental information bottleneck: forcing a model to summarize its understanding back into English text discards approximately 99.96% of its internal representation.

This project proposes a persistent, structured RAM layer. We deploy a swarm of autonomous "Librarian" agents that maintain domain-specific memory buckets, updated asynchronously via OS-level file monitoring.

Our primary research focus is the V2 Latent Space Communication protocol. Instead of exchanging English text, libucks Librarians output raw hidden-state tensors. These latents are aggregated and injected directly into a LoRA-finetuned receiver model. By bypassing the English language entirely, we aim to preserve the high-dimensional context that is typically lost during text decoding.

---

## Quickstart

### Install

```bash
# V1 (Anthropic API strategy)
pip install libucks

# V2 — full latent space pipeline (recommended)
pip install "libucks[latent]"
```

> **V2 hardware:** Apple Silicon (MPS, float16), CUDA, CPU fallback. `Qwen2.5-3B-Instruct` fits in ~6 GB at float16 or ~2 GB at 4-bit NF4.

### Initialize + Train in One Shot

```bash
# Index the repo and train the full V2 pipeline (no API key required)
libucks init --local /path/to/your/repo --train --no-teacher --epochs 5

# With an Anthropic API key — uses Claude Haiku to generate richer Q&A training targets
libucks init --local /path/to/your/repo --train --epochs 5
```

`--train` runs both training phases automatically after indexing:
- **Phase 1 — CommunicationAdapter:** aligns Librarian latent tensors to the Base receiver's embedding space → `adapter.pt`
- **Phase 2 — LoRA Receiver:** fine-tunes `Qwen2.5-3B-Base` to decode framed latent injections into English → `lora_receiver.pt`

### Or Run the Steps Manually

```bash
# 1. Index
libucks init --local /path/to/your/repo

# 2. Train (both phases)
libucks train-adapter --repo /path/to/your/repo --no-teacher --train-receiver --epochs 10

# 3. Re-train receiver only (after a major refactor)
libucks train-adapter --repo /path/to/your/repo --receiver-only --epochs 15
```

### Start the MCP Server

```bash
LIBUCKS_REPO_PATH=/path/to/your/repo libucks serve
```

### Wire Claude Code

Add to `.mcp.json` at your project root:

```json
{
  "mcpServers": {
    "libucks": {
      "command": "/path/to/.venv/bin/libucks",
      "args": ["serve"],
      "env": {
        "LIBUCKS_REPO_PATH": "/path/to/your/repo"
      }
    }
  }
}
```

### Query from the Terminal (No Server Required)

```bash
libucks query --repo /path/to/your/repo "How does the authentication middleware work?"
```

### Keep Memory Fresh (Git Hooks)

```bash
libucks install-hooks --repo /path/to/your/repo
```

Every `git commit` now triggers an incremental bucket update over a Unix socket. No polling, no file watching, no background daemon consuming CPU.

---

## The Black Magic

### V2: Latent Space Communication — Bypassing English Entirely

V1 has a fundamental information bottleneck. Every Librarian-to-Translator exchange is a round-trip through English text:

```
Librarian → reason() → English string → Translator → synthesize → English string
```

A hidden state in a 3B parameter model carries ~40,000 bits of information per position. A token carries ~15 bits. Every English round-trip throws away **99.96% of the model's internal representation**.

**V2 eliminates the intermediate text encoding.**

Librarians return raw `torch.Tensor` hidden states from `Qwen2.5-3B-Instruct`. The Translator is the only component that ever decodes — and it decodes using a LoRA-finetuned `Qwen2.5-3B-Base` **receiver model**, not the Instruct model.

**Why Base, not Instruct, for the receiver?** The Instruct model was RLHF'd on ChatML templates. Injecting arbitrary continuous vectors into it causes "format repair" hallucinations. The Base model has no such conditioning. LoRA on `q_proj`/`v_proj` (rank 4, ~2M trainable params) teaches it to read framed latent injections as meaningful input.

**The injection protocol:**

```
inputs_embeds = [e(<bop>), h_1, ..., h_K, e(<eop>), query_tokens, answer_tokens]
```

`<bop>` and `<eop>` recycle Qwen's native `<|im_start|>` / `<|im_end|>` tokens. No vocabulary resize. No new embeddings. The frame looks structurally identical to what the model has processed billions of times. The LoRA delta teaches it what the latents mean.

**Training objective:**

```
L_total = L_task − λ_sep · L_sep

L_task  = CrossEntropy(generated_tokens | framed_latents)   # teacher forcing
L_sep   = JSD(logits_correct_latent ‖ logits_wrong_latent)  # separation signal
```

Query dropout (50% of steps train without the query prefix) forces the receiver to decode from the latent alone — preventing the model from ignoring the latent entirely and collapsing to memorised Q→A mappings.

**Curriculum mixing** bridges the token and latent manifolds during training:

```
H^(r) = [token_embeds_1..⌊r·K⌋] ⊕ [latents_⌊r·K⌋+1..K]    r ~ U[0,1]
```

Ablation: removing curriculum mixing drops decode success from **70% → 33%**.

---

### The CommunicationAdapter

Aggregates N variable-length Librarian tensors into a fixed `(K=32, D)` soft-prompt before injection:

1. **Attentive Pooling** — a learned query vector cross-attends over each Librarian's token positions → one `(D,)` summary per Librarian.
2. **Inter-Librarian Self-Attention** — 2-layer, 8-head self-attention over N summaries captures cross-bucket relationships.
3. **Output Projection** — K learned queries cross-attend over refined summaries → `(32, D)` soft-prompt.

~2M trainable parameters. Every backbone weight is frozen.

---

### Git-Hook Driven Updates — Zero AI Cost Per Commit

The **Watchdog** is a pure-Python process with zero AI inference. It never stalls, never OOMs, and can be restarted independently of every other component.

`libucks serve` does **not** start the Watchdog. The primary update path is:

```
git post-commit hook
      │
      ▼
Unix socket → StartupRecovery.run()
      │
      ▼
DiffExtractor: git diff HEAD → structured DiffHunk objects
      │
      ▼
CentralAgent: embed added lines → cosine route → UpdateEvent → Librarian
```

Renames are detected and converted directly from the unified diff — no ghost context from delete+create pairs. The `git_sha` embedded in every `ChunkMetadata` record prevents tombstoning chunks that were already updated by a subsequent write.

---

### AST-Parsed Module-Affinity Clustering

INIT doesn't use naive k-means on raw embeddings. It builds a **module-affinity distance matrix** from structural code relationships and feeds it into scipy's agglomerative hierarchical clustering.

The affinity score between any two chunks:

```python
affinity(i, j) = clip(
    cosine_sim(embed_i, embed_j)
    + 0.4  # if same_source_file
    + 0.2, # if file_A_imports_file_B_stem (or vice versa)
    0, 1
)
```

Import detection uses `ast.parse` + `ast.walk` — no subprocess, no language server. This keeps logically coupled code in the same bucket even when surface-text embeddings diverge. A `middleware.py` chunk that imports `jwt_utils` stays co-located with `jwt_utils.py` even if they describe syntactically different operations.

**`ContextCondenser`** — a zero-inference, pure-AST component — produces a token-budget-safe digest for each INIT prose-generation call. Priority: module docstrings → function/class signatures → body lines → hard truncation at 200 tokens. The encoder's 256-token hard limit is never exceeded.

---

### Bucket Mitosis — Self-Organizing Memory

When a bucket exceeds its token threshold (`mitosis_threshold`, default 20,000 tokens), **MitosisService** splits it automatically:

1. Acquire per-bucket write lock. Set `is_splitting = True` in the registry.
2. Re-embed all chunks. Run k-means (k=2).
3. Create two child buckets. Generate a domain label for each via `strategy.reason()`.
4. Instantiate two new Librarians. Remove parent from registry. Register children.
5. Drain the retry buffer — queued `UpdateEvent` objects for the parent are re-routed against the updated registry.

**Invariant:** `len(child_A.chunks) + len(child_B.chunks) == len(parent.chunks)`. No chunk is ever lost.

---

## Architecture

```
┌───────────────────────────────────────┐
│         CODING AGENT (Claude)         │
│   libucks_query("how does X work?")   │
└──────────────────┬────────────────────┘
                   │ stdio / MCP
         ┌─────────▼─────────┐
         │     MCP BRIDGE    │
         └─────────┬─────────┘
                   │
         ┌─────────▼─────────┐
         │    TRANSLATOR     │  ← ONLY natural language output
         └─────────┬─────────┘
                   │ List[Representation]  (tensors in V2)
         ┌─────────▼─────────┐
         │   CENTRAL AGENT   │  ← cosine router over all centroids
         └──┬─────┬──────┬───┘
            │     │      │
       ┌────▼─┐ ┌─▼──┐ ┌─▼────┐
       │Lib A │ │Lib B│ │Lib C │  ... N Librarians
       └──────┘ └─────┘ └──────┘
                   │
         ┌─────────▼─────────┐
         │   BUCKET STORE    │  ← .libucks/buckets/*.md
         └─────────┬─────────┘
                   │
         ┌─────────▼─────────┐
         │     WATCHDOG      │  ← OS events + git diff, zero AI
         └─────────┬─────────┘
                   │
         ┌─────────▼─────────┐
         │ TARGET REPOSITORY │
         └───────────────────┘
```

| Component | Role |
|---|---|
| **CentralAgent** | Embedding-based router. Writes to `BucketRegistry`. Coordinates mitosis. |
| **Librarian** | Per-bucket async event loop. The only agent that writes its `.md` file. |
| **Translator** | The only component that calls `decode()` and outputs natural language. |
| **MCP Bridge** | Exposes `libucks_query` and `libucks_status` over stdio transport. |
| **Bucket** | Markdown file with YAML front-matter. Title-boosted centroid. Chunk provenance via `git_sha`. |

**Strategy is switchable at config time.** Set `strategy = "text"` in `.libucks/config.toml` for V1 (Anthropic API). Set `strategy = "latent"` for the full V2 pipeline. Zero changes to routing, storage, or MCP code.

---

## Configuration

`.libucks/config.toml` — lives inside the target repository (gitignored):

```toml
[model]
strategy           = "latent"           # "text" (V1) | "latent" (V2)
local_model        = "Qwen/Qwen2.5-3B-Instruct"
base_model         = "Qwen/Qwen2.5-3B"
device             = "mps"              # "auto" | "cpu" | "cuda" | "mps"
quantization       = "none"             # "none" | "4bit" | "8bit"
anthropic_model    = "claude-haiku-4-5-20251001"

[routing]
novelty_threshold  = 0.35
top_k              = 3
mitosis_threshold  = 20000
init_bucket_size   = 2000

[paths]
bucket_dir         = ".libucks/buckets"
registry_file      = ".libucks/registry.json"
```

All fields have sane defaults. An empty or missing config file is valid.

---

## CLI Reference

| Command | Description |
|---|---|
| `libucks init --local <path> [--train] [--no-teacher] [--epochs N]` | Index a repo. Optionally train in one shot. |
| `libucks train-adapter --repo <path> [--no-teacher] [--train-receiver] [--receiver-only] [--epochs N]` | Train adapter and/or LoRA receiver. |
| `libucks query --repo <path> "question"` | Run a query directly. Bypasses MCP, no timeout. |
| `libucks serve` | Start the MCP server over stdio. |
| `libucks install-hooks --repo <path>` | Append git post-commit hooks. Never overwrites. |
| `libucks use <path>` | Set the active repo for `libucks serve`. |

---

## License

MIT. See [LICENSE](./LICENSE).

---

## Contributing

Issues and PRs are welcome. Before opening a PR:

- Run `pytest tests/unit/` — all tests must be green.
- Read `ARCHITECTURE.md` §4 (Latent Space Interface Constraint) before touching anything in `libucks/thinking/`.
- The `Translator` is the **only** component permitted to call `decode()`. This boundary is non-negotiable.

---

*`libucks` — because your agent shouldn't be reading the same file for the hundredth time.*
