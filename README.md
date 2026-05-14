![libucks Architecture Banner](./docs/banner.png)

# libucks — Librarian Buckets

**A persistent, latent-space memory server for coding agents. Your agent stops reading files. It queries memory.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![MCP Compatible](https://img.shields.io/badge/MCP-compatible-green.svg)](https://modelcontextprotocol.io/)

---

## The Brutal Reality

Coding agents working on large repositories are broken by design. Three compounding failure modes kill them in production:

1. **Context bloat.** 100,000+ line repos do not fit in a context window. Agents that try to read everything either truncate silently or blow the token budget on the first query.

2. **"Lost in the middle" degradation.** LLM attention is not uniform. Content buried at position 60K in a 128K context window gets ~40% less weight than content at the boundaries. Your critical auth logic is functionally invisible.

3. **Runaway API cost.** Re-reading unchanged files on every query burns money on redundant computation. A file unchanged since yesterday does not need to be re-read today.

Every existing solution — RAG pipelines, context compression, file summarizers — patches one failure mode while ignoring the others. They are also stateless: no memory between queries, no awareness of code evolution, no understanding of domain boundaries.

**`libucks` is not a patch.**

---

## Enter libucks

`libucks` is a **persistent, structured memory layer** between the coding agent and the repository. It maintains a swarm of domain-specific **context buckets**, updates them asynchronously as code changes, and serves compressed, semantically-routed context to the agent via the **Model Context Protocol (MCP)**.

The agent is given a single tool: `libucks_query(query, top_k=3)`. It never reads raw files again.

---

## Research Foundations

libucks V2 is grounded in five papers from the agent communication and latent reasoning literature:

| Paper | arXiv | Role in libucks |
|---|---|---|
| **Interlat** — *Enabling Agents to Communicate Entirely in Latent Space* | [2511.09149](https://arxiv.org/abs/2511.09149) | **Primary architecture.** The latent injection protocol, `<bop>`/`<eop>` framing, L_task + L_sep training objective, and query dropout are all direct implementations of Interlat §3.2-3.3 and §10. |
| **Coconut** — *Training Large Language Models to Reason in a Continuous Latent Space* | [2412.06769](https://arxiv.org/abs/2412.06769) | Motivates reasoning without language constraints. The curriculum mixing schedule (token→latent interpolation) adapts Coconut's chain-of-continuous-thought approach to the cross-agent communication setting. |
| **AgentPrune** — *Cut the Crap: An Economical Communication Pipeline for LLM-Based Multi-Agent Systems* | [2410.02506](https://arxiv.org/abs/2410.02506) | Informs the bucket routing design. Sparse, structured communication between agents (Librarians → Translator) rather than broadcasting full context to all agents. |
| **Latent Collaboration in Multi-Agent Systems** | [2511.20639](https://arxiv.org/abs/2511.20639) | Supports the multi-Librarian → single Translator aggregation pattern via the CommunicationAdapter's inter-agent attention layers. |
| **Reasoning Models Don't Always Say What They Think** | [2505.05410](https://arxiv.org/abs/2505.05410) | Motivates the architectural constraint that Librarians never produce natural language. Faithfulness is enforced structurally: Librarians return tensors, not strings. |

The research papers are available in full in [`docs/`](./docs/).

---

## Quickstart

### Install

```bash
# V1 (Anthropic API strategy — no local model required)
pip install libucks

# V2 — full latent space pipeline (recommended)
pip install "libucks[latent]"
```

> **V2 hardware:** Apple Silicon (MPS, float16), CUDA, CPU fallback. Current PoC uses `Qwen2.5-0.5B` (encoder) and `Qwen2.5-0.5B-Base` (receiver) — fits comfortably in 4 GB. Production target is `Qwen2.5-1.5B` or `Qwen2.5-3B`.

### Initialize + Train in One Shot

```bash
# Index the repo and train the full V2 pipeline (no API key required)
libucks init --local /path/to/your/repo --train --no-teacher

# With an Anthropic API key — uses Claude Haiku to generate richer Q&A training targets
libucks init --local /path/to/your/repo --train
```

`--train` runs both training phases automatically after indexing:
- **Phase 1 — CommunicationAdapter:** aligns Librarian latent tensors to the Base receiver's embedding space → `adapter.pt`
- **Phase 2 — LoRA Receiver:** fine-tunes `Qwen2.5-0.5B-Base` with rank-16 LoRA on `q_proj`/`v_proj` to decode framed latent injections into English → `lora_receiver.pt`

### Or Run the Steps Manually

```bash
# 1. Index
libucks init --local /path/to/your/repo

# 2. Train (both phases)
libucks train-adapter --repo /path/to/your/repo --no-teacher --train-receiver --epochs 5

# 3. Re-train receiver only (after a major refactor)
libucks train-adapter --repo /path/to/your/repo --receiver-only --epochs 5
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
libucks query "How does the authentication middleware work?" --repo /path/to/your/repo
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

A hidden state in a 0.5B parameter model carries thousands of bits of information per position. A token carries ~15 bits. Every English round-trip throws away the vast majority of the model's internal representation.

**V2 eliminates the intermediate text encoding.**

Librarians return raw `torch.Tensor` hidden states from `Qwen2.5-0.5B-Instruct`. The Translator is the only component that ever decodes — and it decodes using a LoRA-finetuned `Qwen2.5-0.5B-Base` **receiver model**, not the Instruct model.

**Why Base, not Instruct, for the receiver?** The Instruct model was RLHF'd on ChatML templates. Injecting arbitrary continuous vectors into it causes "format repair" hallucinations. The Base model has no such conditioning. LoRA on `q_proj`/`v_proj` (rank 16, ~2M trainable params) teaches it to read framed latent injections as meaningful input.

**The injection protocol** (Interlat §3.2):

```
inputs_embeds = [e(<bop>), h_1, ..., h_K, e(<eop>), query_tokens, answer_tokens]
```

`<bop>` and `<eop>` recycle Qwen's native `<|im_start|>` / `<|im_end|>` tokens. No vocabulary resize. No new embeddings. The frame looks structurally identical to what the model has processed billions of times. The LoRA delta teaches it what the latents mean.

**Training objective** (Interlat §3.3):

```
L_total = L_task − λ_sep · L_sep

L_task  = CrossEntropy(generated_tokens | framed_latents)   # teacher forcing
L_sep   = JSD(logits_correct_latent ‖ logits_wrong_latent)  # separation signal
```

Query dropout (50% of steps train without the query prefix) forces the receiver to decode from the latent alone — preventing the model from ignoring the latent entirely and collapsing to memorised Q→A mappings. **If `sep` stays at 0.0000 past epoch 3, the latent is being ignored — stop and check query dropout.**

**Curriculum mixing** bridges the token and latent manifolds during training:

```
H^(r) = [token_embeds_1..⌊r·K⌋] ⊕ [latents_⌊r·K⌋+1..K]    r ~ U[0,1]
```

---

### The CommunicationAdapter

Aggregates N variable-length Librarian tensors into a fixed `(K=32, D)` soft-prompt before injection:

1. **Attentive Pooling** — a learned query vector cross-attends over each Librarian's token positions → one `(D,)` summary per Librarian.
2. **Inter-Librarian Self-Attention** — 2-layer, 8-head self-attention over N summaries captures cross-bucket relationships.
3. **Output Projection** — K learned queries cross-attend over refined summaries → `(32, D)` soft-prompt.

~2M trainable parameters. Every backbone weight is frozen.

---

### Git-Hook Driven Updates — Zero AI Cost Per Commit

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

Import detection uses `ast.parse` + `ast.walk` — no subprocess, no language server.

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
         │  GIT HOOK SOCKET  │  ← post-commit → Unix socket → StartupRecovery
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

## PoC Status

The current implementation is a working proof of concept with known limitations:

| Component | Status |
|---|---|
| Bucket indexing (init, routing, storage) | ✅ Production-quality |
| Git hook pipeline (commit → re-index) | ✅ Working end-to-end |
| MCP server (libucks_query, libucks_status) | ✅ Working |
| Multi-repo support | ✅ Validated on libucks-self + click |
| CommunicationAdapter training | ✅ Stable (cross-bucket cos ~0.57) |
| LoRA receiver training | ✅ Trains, sep > 0 |
| Answer factual grounding | ⚠️ Hallucination ceiling — 0.5B model + small datasets produce plausible but often wrong answers |
| Mitosis (bucket splitting) | ✅ Implemented, not stress-tested |

The routing and pipeline mechanics are correct. The answer quality limitation is a model capacity / training data problem, not an architectural one. Upgrading to `Qwen2.5-1.5B` or generating 400+ QA pairs per repo are the two paths to grounded answers.

---

## Configuration

`.libucks/config.toml` — lives inside the target repository (gitignored):

```toml
[model]
strategy           = "latent"           # "text" (V1) | "latent" (V2)
local_model        = "Qwen/Qwen2.5-0.5B-Instruct"
base_model         = "Qwen/Qwen2.5-0.5B"
device             = "auto"             # "auto" | "cpu" | "cuda" | "mps"
quantization       = "none"             # "none" | "4bit" | "8bit"
anthropic_model    = "claude-haiku-4-5-20251001"

[routing]
novelty_threshold  = 0.35
top_k              = 3
mitosis_threshold  = 20000
```

All fields have sane defaults. An empty or missing config file is valid.

---

## CLI Reference

| Command | Description |
|---|---|
| `libucks init --local <path> [--train] [--no-teacher] [--epochs N]` | Index a repo. Optionally train in one shot. |
| `libucks train-adapter --repo <path> [--no-teacher] [--train-receiver] [--receiver-only] [--epochs N]` | Train adapter and/or LoRA receiver. |
| `libucks query "question" --repo <path>` | Run a query directly. Bypasses MCP, no timeout. |
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
- LoRA must always be injected with the same rank used during training (currently `r=16`). If you change the rank, you must retrain from scratch.

---

*`libucks` — because your agent shouldn't be reading the same file for the hundredth time.*
