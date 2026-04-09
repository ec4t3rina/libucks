# libucks — Architecture Reference

**Librarian Buckets** | Local AI Memory Server for Coding Agents

---

## 1. Problem Statement

Coding agents operating on large repositories suffer from three compounding failure modes:

1. **Context bloat** — reading 100 000+ lines consumes the entire context window.
2. **"Lost in the middle" degradation** — LLM attention weakens for content far from the prompt boundaries.
3. **Runaway API cost** — re-reading unchanged files on every query is wasteful.

`libucks` acts as a persistent, structured RAM layer between the coding agent and the repository. It maintains a swarm of domain-specific context buckets, updates them asynchronously as code changes, and serves compressed context to the agent via the Model Context Protocol (MCP). The agent never reads raw files directly; it queries the memory server.

---

## 2. System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CODING AGENT (Claude)                        │
│                                                                     │
│   libucks_query("how does auth work?") ──────► MCP Tool Call       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ stdio / MCP protocol
┌──────────────────────────────▼──────────────────────────────────────┐
│                         MCP BRIDGE                                  │
│              (mcp Python SDK, versioned tool schemas)               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │     TRANSLATOR      │  ◄── Only component that
                    │  (Spokesperson)     │       produces natural language
                    └──────────┬──────────┘
                               │ Representation objects (str in V1, tensor in V2)
                    ┌──────────▼──────────┐
                    │   CENTRAL AGENT     │  ◄── Embedding-based router
                    │   (Dispatcher)      │
                    └──┬──────┬──────┬───┘
                       │      │      │
           ┌───────────▼┐  ┌──▼────┐ ┌▼──────────┐
           │ Librarian A│  │Lib. B │ │Librarian C │   ... N librarians
           │ frontend   │  │auth   │ │db-schema   │
           └──────┬─────┘  └───┬───┘ └────┬───────┘
                  │            │           │
           ┌──────▼────────────▼───────────▼───────┐
           │           BUCKET STORE                 │
           │   .libucks/buckets/*.md  (+ registry)  │
           └────────────────────────────────────────┘
                               ▲
                    ┌──────────┴──────────┐
                    │      WATCHDOG       │  ◄── OS file events + git diff
                    └─────────────────────┘
                               ▲
                    ┌──────────┴──────────┐
                    │   TARGET REPOSITORY │
                    └─────────────────────┘
```

---

## 3. Components

### 3.1 Buckets

**What they are:** Plain Markdown files stored under `.libucks/buckets/` inside the target repository. Each file is a highly condensed, domain-specific slice of context written by a Librarian.

**File format:**

```markdown
---
bucket_id: "a3f8c2d1"
domain_label: "authentication middleware"
centroid_embedding: "<base64-encoded float32 array>"
token_count: 1842
chunks:
  - chunk_id: "c001"
    source_file: "src/auth/middleware.py"
    start_line: 12
    end_line: 47
    git_sha: "e4f9a3b"
    token_count: 312
  - ...
---

## Authentication Middleware

The auth middleware validates JWTs on every inbound request. Tokens are
signed with RS256. Expiry is enforced at 15 minutes for access tokens and
7 days for refresh tokens. The `require_role` decorator gates endpoints...
```

**Chunk metadata** (`source_file`, `start_line`, `end_line`, `git_sha`) is the authoritative provenance record. It enables precise invalidation when code changes without re-reading the whole bucket.

**`domain_label`:** An LLM-generated 5–10 word semantic title (e.g. `"CLI decorator binding and command registration"`), produced by `strategy.reason()` during INIT. It replaces the old mechanical file-stem label and is used to anchor the centroid (see §3.3).

**Token limit:** Configurable, default **20 000 tokens**. Exceeding this threshold triggers Bucket Mitosis (§3.4).

---

### 3.2 Watchdog

**What it is:** A lightweight, non-AI Python process using the `watchdog` library to listen for OS-level file system events on the target repository.

**What it does:**
1. On `FileModifiedEvent` for a tracked source file, it calls `DiffExtractor`.
2. `DiffExtractor` runs `git diff HEAD -- <filepath> --find-renames` and parses the output into structured `DiffHunk` objects (added lines, removed lines, line ranges).
3. Renames are detected and converted to `RenameEvent` objects rather than delete+create pairs (preventing ghost context — see §5.2).
4. The resulting `DiffEvent` is placed onto the Central Agent's async input queue.

**Design constraint:** The Watchdog performs zero AI inference. It is a pure data-extraction process. This keeps it fast, always-on, and independently restartable.

---

### 3.3 Central Agent (Router / Dispatcher)

The Central Agent is the embedding-based brain of the UPDATE and QUERY workflows. It runs as an async event loop and is the only component that writes to `BucketRegistry`.

**Routing algorithm (UPDATE):**

1. Embed the diff's added lines: `q = embed(added_content)` → L2-normalized vector in R^384 (default: `all-MiniLM-L6-v2`).
2. For each bucket `b` in `BucketRegistry`: `similarity(q, centroid_b) = dot(q, centroid_b)`.
3. Sort descending. Select top-K buckets where `similarity ≥ (1 − novelty_threshold)`.
4. If top-1 similarity falls below the novelty threshold → emit `CreateBucketEvent` (new domain detected).

**Routing algorithm (QUERY):** Same cosine similarity, but read-only — no lock acquisition.

**Novelty threshold:** Default `0.35` cosine distance (`= 0.65` cosine similarity). Configurable per repo in `.libucks/config.toml`.

**Centroid maintenance (title-boosted):** After every Librarian write, the centroid is recomputed as:

```
centroid = normalize(
    0.8 * mean(embed(chunk) for chunk in bucket.chunks)
  + 0.2 * embed(domain_label)
)
```

The 20% title contribution anchors the bucket's vector identity to its semantic purpose. Without it, centroid drift toward surface-text noise is unchecked. The blend weight α=0.2 was chosen so the title steers direction without overriding the empirical chunk distribution.

**Mitosis coordination:** The Central Agent maintains an `is_splitting` flag per bucket in `BucketRegistry`. If set, incoming `UpdateEvent` objects for that bucket are held in a retry buffer (3 retries, 100ms backoff) and re-routed after mitosis completes.

**Tombstone dispatch:** For removed lines in a diff, the Central Agent identifies overlapping `ChunkMetadata` records and emits `TombstoneEvent(chunk_ids, bucket_ids)` to the relevant Librarians.

---

### 3.4 Librarians

Each Librarian is an async event loop bound to exactly one Bucket. It is the only agent that writes to its bucket's `.md` file.

**Responsibilities:**

| Event | Action |
|---|---|
| `UpdateEvent` | Re-read bucket → call `ThinkingStrategy.reason(diff, context)` → write updated prose → recompute centroid → update `BucketRegistry` |
| `TombstoneEvent` | Remove stale `ChunkMetadata` entries from front-matter → re-render prose without purged chunks |
| `QueryEvent` | Call `ThinkingStrategy.reason(query, context)` → return `Representation` to `QueryOrchestrator` |
| `InitEvent` | Accept initial `RawChunk` list → call `ThinkingStrategy.reason` to produce condensed prose → write bucket |

**Concurrency:** All write operations acquire the per-bucket `asyncio.Lock` stored in `BucketRegistry`. Read operations (query) do not acquire the lock.

**Mitosis trigger:** After an `UpdateEvent` write, if `token_count > mitosis_threshold`, the Librarian signals `MitosisService`.

---

### 3.5 Mitosis Service

Triggered when a bucket exceeds its token limit. Runs as an isolated async task.

**Process:**
1. Acquire per-bucket write lock + set `is_splitting = True` in `BucketRegistry`.
2. Re-embed all chunks in the bucket.
3. Run k-means clustering with k=2 on chunk embeddings.
4. Create two new child buckets via `BucketStore`. Generate a domain label for each via `ThinkingStrategy.reason`.
5. Instantiate two new Librarians, one per child bucket.
6. Remove parent from `BucketRegistry`. Register children.
7. Release write lock + clear `is_splitting`. The Central Agent drains the retry buffer for the parent bucket ID, re-routes each event against the updated registry.

**Invariant:** `len(child_A.chunks) + len(child_B.chunks) == len(parent.chunks)`. No chunk may be lost during mitosis.

---

### 3.6 Translator (Spokesperson)

**The only component in the system that produces natural language output.**

Receives N `Representation` objects from N Librarians plus the original query string. Calls `ThinkingStrategy.decode` and then synthesizes a single coherent English answer via one final `ThinkingStrategy.reason` call. Returns a plain `str` to `MCPBridge`.

**Why this isolation matters:** In V2, Librarians will communicate in Latent Space (raw tensors). The Translator is the decode boundary — it is the only place where tensors are converted back to text. No other component is permitted to call a text-generation model and return natural language.

---

### 3.7 MCP Bridge

An MCP server implemented with the `mcp` Python SDK, started by `libucks serve` over stdio transport.

**Registered tools:**

| Tool | Signature | Purpose |
|---|---|---|
| `libucks_query` | `(query: str, top_k: int = 3) -> str` | Primary context query |
| `libucks_status` | `() -> dict` | System health: bucket count, token totals, last-updated timestamps |

Tool schemas are defined in a versioned `tools_v1.json` manifest loaded at startup. The schema is decoupled from the Python implementation to allow MCP spec evolution without internal refactoring.

---

### 3.8 AspectMapper

**File:** `libucks/parsing/aspect_mapper.py`

A pure-Python component (no AI, no torch) that produces a module-affinity distance matrix for use during INIT clustering. Replaces raw cosine distance with a richer signal that keeps logically related chunks together.

**Affinity score between chunk i and chunk j:**

```
affinity(i, j) = clip(cosine_sim(embed_i, embed_j)
                     + same_file_bonus(i, j)      # +0.4 if same source file
                     + import_link_bonus(i, j),   # +0.2 if file A imports file B's stem
                     0, 1)
```

**Import detection:** Uses `ast.parse` on each Python chunk's content to extract imported module stems. A bonus applies if either file's stem appears in the other's imports.

**Clustering:** Converts affinity to distance (`1 - affinity`), then calls `scipy.cluster.hierarchy.linkage` with `method="average"`. The same agglomerative algorithm as before — only the distance matrix changes.

---

### 3.9 ContextCondenser

**File:** `libucks/thinking/context_condenser.py`

A pure-Python component that produces a token-budget-safe digest of source code for the INIT prose-generation LLM call. The Encoder (`all-MiniLM-L6-v2`) has a hard 256-token limit; the condenser ensures the digest fits within 200 tokens, leaving headroom for the query prefix.

**Interface:**
```python
ContextCondenser().condense(chunks: List[RawChunk], budget_tokens: int = 200) -> str
```

**Priority order (greedy fill):**
1. Module-level docstring (file overview — highest signal density)
2. Function/class signatures + their docstrings (no body lines)
3. Truncated body lines if budget remains
4. Hard truncation at the token boundary

**No model dependency.** Uses only `ast.parse` and `ast.unparse`. Safe to call from any context.

---

## 4. The Latent Space Interface Constraint (V1 → V2 Migration Contract)

> **This section is a hard architectural constraint. All contributors must read it.**

### The Problem V2 Solves

In V1, every Librarian-to-Translator communication is a round-trip through English text:
```
Librarian → reason(query, context) → English string → Translator → synthesize → English string
```

This is correct but slow. V2 eliminates the intermediate English encoding by having Librarians return raw hidden-state tensors from a local open-source model. The Translator is the only component that runs a decode head to convert tensors → text.

### The Strategy Interface

To make V1 and V2 structurally identical at every call site, all Librarian reasoning is mediated through the `ThinkingStrategy` abstract base class:

```
ThinkingStrategy (ABC)
│
├── encode(text: str) -> Representation
│     V1: returns text unchanged (str)
│     V2: runs text through local model encoder → hidden state tensor
│
├── reason(query: str, context: str) -> Representation
│     V1: constructs prompt → async Ollama call → returns response str
│     V2: encodes query + context → model forward pass in latent space → returns tensor
│
└── decode(result: Representation) -> str
      V1: returns result unchanged (it's already a str)
      V2: runs decoder head on tensor → returns English str

Representation = Union[str, torch.Tensor]  # type alias
```

**Implementations:**
- `TextStrategy` — V1. Fully functional. Uses async `httpx` calls to a local Ollama daemon.
- `LatentStrategy` — V2 stub. Every method raises `NotImplementedError` with a message pointing to the V2 upgrade ticket. It exists now so the interface is exercised in tests from day one.

### The Rule

> **No component other than the Translator may call `ThinkingStrategy.decode` and return its result as the final output to the MCP Bridge.**

Librarians call `encode` and `reason`. The Translator calls `reason` (to synthesize) and `decode` (to produce the final string). This boundary must be enforced in code review.

---

## 5. Workflows

### 5.1 INIT

```
libucks init --local <repo-path>
      │
      ▼
ASTParser (Tree-sitter + GrammarRegistry)
  - Walk all source files (skips hidden dirs, noise dirs, files > 500 KB)
  - Extract function/class definitions; fallback to 50-line windows
  - Produce: List[RawChunk(source_file, start_line, end_line, content, language)]
      │
      ▼
EmbeddingService.embed_batch(all_chunk_contents)  → (N, 384) L2-normalised
      │
      ▼
AspectMapper.cluster(chunks, embeddings, n_clusters)
  - n_clusters = max(1, min(total_tokens // init_bucket_size, N))
  - Affinity matrix = cosine_sim + same-file bonus(+0.4) + import-link bonus(+0.2)
  - Agglomerative clustering (scipy, method="average") on 1-affinity distance
  - Each cluster → one initial Bucket
      │
      ▼
For each cluster  [asyncio.gather — all LLM calls run concurrently]:
  ContextCondenser.condense(cluster_chunks, budget=200 tokens) → digest
  strategy.reason(prompt, digest) → "TITLE: <title>\n---\n<prose>"
  Parse → (title, prose)
  title_embed = EmbeddingService.embed(title)
  centroid = normalize(0.8 * chunk_mean + 0.2 * title_embed)
  BucketStore.create(domain_label=title, centroid=blended, prose=prose)
  BucketRegistry.register(bucket_id, centroid, token_count)
      │
      ▼
BucketRegistry.save()  →  .libucks/registry.json
      │
      ▼
libucks serve  (ready to accept MCP connections)
```

**Grammar management:** `GrammarRegistry` maps file extensions to language names and lazily downloads compiled Tree-sitter grammar `.so` binaries from the tree-sitter GitHub releases API on first encounter, caching under `~/.libucks/grammars/`. No grammars are bundled at install time.

**INIT concurrency:** All `strategy.reason()` calls for prose generation are issued with `asyncio.gather`, so N API calls run concurrently rather than serially. INIT time scales with the slowest single call, not the sum.

---

### 5.2 UPDATE

```
OS file save event (Ctrl+S)
      │
      ▼
WatchdogService detects FileModifiedEvent
      │
      ▼
DiffExtractor
  - git diff HEAD -- <file> --find-renames
  - Parse unified diff → List[DiffHunk]
  - Renames → RenameEvent (NOT delete+add)
      │
      ▼
asyncio.Queue  ──────►  CentralAgent event loop
      │
      ├── For added lines:
      │     embed → cosine similarity → top-K buckets
      │     if top-1 < novelty_threshold → CreateBucketEvent
      │     else → UpdateEvent(bucket_id, diff_hunk) → Librarian queue
      │
      ├── For removed lines:
      │     find overlapping ChunkMetadata by (source_file, line_range, git_sha)
      │     → TombstoneEvent(chunk_ids) → Librarian queue
      │
      └── For renames:
            → PathUpdateEvent(old_path, new_path) → all Librarians
              (updates source_file field; no content purge)
```

**Ghost context prevention:** Every `ChunkMetadata` record carries `git_sha`. A tombstone pass checks `chunk.git_sha` against the deletion commit. If the sha is newer than the deletion (i.e., the chunk was already updated by a subsequent write), the chunk is not purged — it survived the deletion.

---

### 5.3 QUERY

```
Claude Code calls libucks_query("how does auth work?")
      │
      ▼
MCPBridge receives MCP tool call
      │
      ▼
QueryOrchestrator
  1. embed(query) → q
  2. cosine similarity vs all centroids → top-K bucket_ids (read-only)
  3. asyncio.gather → QueryEvent(query, bucket_id) to each Librarian
      │
      ▼
Each Librarian (concurrently):
  ThinkingStrategy.reason(query, bucket_content) → Representation
      │
      ▼
Translator receives List[Representation] + original query
  ThinkingStrategy.reason(synthesize N partial results)
  ThinkingStrategy.decode(result) → str
      │
      ▼
MCPBridge wraps str in MCP tool response schema → Claude Code
```

---

## 6. Data Models

```
BucketFrontMatter
  bucket_id:          str
  domain_label:       str
  centroid_embedding: bytes   # base64 float32 array
  token_count:        int
  chunks:             List[ChunkMetadata]

ChunkMetadata
  chunk_id:     str
  source_file:  str
  start_line:   int
  end_line:     int
  git_sha:      str
  token_count:  int

RawChunk
  source_file:  str
  start_line:   int
  end_line:     int
  content:      str
  language:     str

DiffHunk
  file:         str
  old_start:    int
  old_end:      int
  new_start:    int
  new_end:      int
  added_lines:  List[str]
  removed_lines: List[str]

Events
  DiffEvent(file, hunks, is_rename, old_path, new_path)
  UpdateEvent(bucket_id, hunk)
  TombstoneEvent(chunk_ids, bucket_ids)
  PathUpdateEvent(old_path, new_path)
  QueryEvent(query, bucket_id)
  CreateBucketEvent(seed_content)
```

---

## 7. Configuration

`.libucks/config.toml` (lives inside the target repository, gitignored):

```toml
[model]
anthropic_model    = "claude-haiku-4-5-20251001"
embedding_model    = "all-MiniLM-L6-v2"
local_model        = "Qwen/Qwen2.5-3B-Instruct"   # V2 latent strategy only
quantization       = "none"                         # "none" | "4bit" | "8bit"
device             = "auto"                         # "auto" | "cpu" | "cuda" | "mps"
strategy           = "text"                         # "text" | "latent"

[routing]
novelty_threshold  = 0.35    # cosine distance below which a new bucket is created
top_k              = 3       # number of buckets queried per request
mitosis_threshold  = 20000   # token count at which a live bucket is eligible for manual mitosis
init_bucket_size   = 2000    # target raw-token count per bucket at INIT; controls seeding density

[paths]
bucket_dir         = ".libucks/buckets"
registry_file      = ".libucks/registry.json"
pending_events     = ".libucks/pending_events.jsonl"
log_file           = ".libucks/libucks.log"
grammar_cache      = "~/.libucks/grammars"
repo_cache         = "~/.libucks/repos"
```

---

## 8. Directory Layout

```
libucks/                         ← this repository (the libucks tool itself)
├── main.py                      ← CLI entry point (click group)
├── pyproject.toml
├── tools_v1.json                ← versioned MCP tool schema manifest
├── ARCHITECTURE.md
├── IMPLEMENTATION_PLAN.md
│
├── libucks/
│   ├── config.py
│   ├── models/
│   │   ├── bucket.py            ← BucketFrontMatter, Bucket
│   │   ├── chunk.py             ← ChunkMetadata, RawChunk
│   │   └── events.py            ← all Event dataclasses
│   ├── storage/
│   │   ├── bucket_store.py      ← CRUD for .md files with YAML front-matter
│   │   └── bucket_registry.py   ← in-memory index + per-bucket asyncio.Lock
│   ├── embeddings/
│   │   └── embedding_service.py ← sentence-transformers singleton wrapper
│   ├── thinking/
│   │   ├── base.py              ← ThinkingStrategy ABC, Representation type alias
│   │   ├── text_strategy.py     ← V1: async Anthropic API
│   │   ├── latent_strategy.py   ← V2: Qwen encoder + LoRA receiver
│   │   ├── context_condenser.py ← §3.9 token-budget-safe digest (pure AST, no model)
│   │   ├── communication_adapter.py ← N Librarian tensors → fixed (K, D) soft-prompt
│   │   ├── compressor.py        ← (L, D) → (K, D) latent compression
│   │   └── model_manager.py     ← Qwen load/cache/unload singleton
│   ├── parsing/
│   │   ├── ast_parser.py        ← tree-sitter → RawChunk list
│   │   ├── grammar_registry.py  ← lazy grammar download + cache
│   │   └── aspect_mapper.py     ← §3.8 affinity graph + agglomerative clustering
│   ├── diff/
│   │   └── diff_extractor.py    ← git diff → DiffHunk list, rename detection
│   ├── watchdog_service.py
│   ├── central_agent.py         ← async routing loop
│   ├── librarian.py             ← per-bucket async event loop
│   ├── mitosis.py               ← split logic
│   ├── translator.py            ← ONLY natural language output producer
│   ├── mcp_bridge.py            ← MCP server
│   ├── query_orchestrator.py
│   ├── init_orchestrator.py
│   └── health_monitor.py
│
└── tests/
    ├── conftest.py
    ├── fixtures/
    │   ├── routing/
    │   │   └── needle_cases.json
    │   └── repos/
    │       └── sample_repo/      ← small fixture Python project
    ├── unit/
    └── integration/
```

---

## 10. V2.1 Interlat-Lite: Stable Latent Injection

### Problem with V2.0 (NormMatch + Residual Anchoring)

The V2.0 decode path achieved 0.0002 MSE loss during adapter training but
produced character-soup at inference. Three root causes:

1. **MSE optimises geometry, not decodability.** A vector can be geometrically
   close to a target hidden state but still off the language manifold —
   deviations compound exponentially through autoregressive generation.
2. **Instruct model ChatML rigidity.** Qwen2.5-3B-Instruct was RLHF'd on
   ChatML templates. Injecting arbitrary continuous vectors causes the model
   to "repair" a perceived format corruption, yielding BPE fragments.
3. **The receiver was never trained.** Latents were injected into a *frozen*
   model. No gradient ever connected "adapter output → coherent generated text."

### Model Split (V2.1)

| Role | Model | Purpose |
|---|---|---|
| **Encoder** | Qwen2.5-3B-Instruct | Librarians: `encode()` / `reason()` → latent tensors |
| **Receiver** | Qwen2.5-3B-Base + LoRA | Translator: `receive()` → natural language |

Using a Base model eliminates ChatML rigidity. LoRA fine-tuning (q_proj / v_proj
only, r=16, alpha=32) trains the model to interpret framed latent injections.

### Injection Protocol

```
inputs_embeds = [e(<bop>), h_1, ..., h_K, e(<eop>)]   shape: (K+2, D)
```

`CommunicationAdapter.frame(soft_prompt, bop_embed, eop_embed)` prepends
`e(<bop>)` and appends `e(<eop>)` boundary embeddings. The receiver has been
trained to recognise this structure.

No NormMatch. No Residual Anchoring. The trained Base receiver handles the
manifold gap directly.

### Training Objective

```
L_total = L_task - λ_sep * L_sep

L_task = CrossEntropy(generated_tokens | injected_framed_latents)   # teacher forcing
L_sep  = JSD(logits_correct_latent, logits_wrong_latent)            # from losses.py
```

Minimising `L_task` provides the only real decodability signal — the model
must generate the correct target text given the injected latent. Minimising
`-L_sep` (maximising JSD) prevents the model from ignoring the latent signal.

### Curriculum Mixing

At each training step, sample `r ~ U[0, 1]`:

```
H^(r) = [e_1, ..., e_{floor(r·K)}] ⊕ [h_{floor(r·K)+1}, ..., h_K]
          ← token embeddings →        ←        latents          →
```

`r = 0` → pure latents; `r = 1` → pure token embeddings. Uniform sampling
forces the receiver to handle any mixture, bridging the known token manifold
and the target latent manifold smoothly. Removal of curriculum drops success
from 70% to 33% in Interlat ablations.

### Training Pipeline Optimisations (Phase 12.7)

#### Pre-computation cache

All `TextStrategy.reason()` API calls and `LatentStrategy.reason()` local
inference calls are executed **once per bucket** before the epoch loop begins.
Results — `inputs_embeds`, `inputs_embeds_wrong`, `target_ids`, `prefix_len` —
are stored as CPU tensors.  The epoch loop moves each item to the device just
before the gradient step and releases it immediately after.

This eliminates the dominant bottleneck: with 47 buckets and 30 epochs the old
code made 1,410 API + 1,410 local inference calls (≈30 min). The new code makes
47 API + 47 local inference calls in a one-time pre-compute phase (≈2–3 min),
then trains for ~1–2 min of pure GPU work.

#### Batched dual forward pass

`LoRAReceiverTrainer._forward_and_losses()` stacks the correct-latent and
wrong-latent embed sequences into a single `(2, S, D)` tensor:

```
embeds_batch = torch.stack([inputs_embeds, inputs_embeds_wrong], dim=0)
out_batch    = model(inputs_embeds=embeds_batch)   # one GPU call, not two
```

This halves GPU forward-pass overhead per gradient step.

#### L_sep gradient requirement

`logits_tgt` and `logits_wrong_tgt` must **not** be detached before passing to
`separation_loss()`.  Detaching them (as in the original Phase 12.6 code) makes
`sep` a constant in the autograd graph — `total.backward()` propagates only
through `task_loss`, silently zeroing the separation signal.  Without it, the
receiver ignores the latent prefix and falls back to unconditional language
modelling, producing word-salad output.

#### Gradient accumulation

Default `ACCUM_STEPS = 8`.  `accumulate_step(batch, scale=8, step=is_last)`
divides the loss by 8 before each `backward()` call and only advances the
optimizer after every 8th bucket (or at end of epoch).  Effective batch size = 8
without requiring padded batching, keeping peak MPS memory bounded.

#### Recommended hyperparameters

| Param | Phase 12.6 (broken) | Phase 12.7 (fixed) | Rationale |
|---|---|---|---|
| `lora_r` | 32 | 4 | r=32 overfits 47 samples; r=4 sufficient (Hu et al. 2022 Table 2) |
| `lora_alpha` | 64 | 8 | Preserves scaling ratio alpha/r = 2 |
| `lr` | 1e-3 | 2e-4 | Standard LoRA fine-tuning range (1e-4 – 3e-4) |

### Implementation Files

| File | Role |
|---|---|
| `libucks/thinking/curriculum.py` | `CurriculumMixer.mix(latents, tok_embeds, r)` |
| `libucks/thinking/communication_adapter.py` | `frame(soft_prompt, bop, eop)` + 8-head MHA |
| `libucks/thinking/model_manager.py` | `load_base_model()` / `get_base_model()` |
| `libucks/thinking/training/losses.py` | `separation_loss(logits_c, logits_w)` |
| `libucks/thinking/training/lora_trainer.py` | `LoRAReceiverTrainer` (train_step, accumulate_step) |
| `libucks/thinking/latent_strategy.py` | `receive(framed_latent)` — new decode path |
| `scripts/train_lora_receiver.py` | Pre-compute cache + gradient accumulation loop |

### Latent Space Constraint (unchanged)

Librarians only produce `Representation` objects. ONLY the Translator's
`receive()` path (which calls the LoRA Base receiver) outputs natural language.

---

## 9. Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| `pydantic` | `>=2.0` | Data model validation |
| `sentence-transformers` | `>=3.0` | Local embedding model |
| `numpy` | `>=1.26` | Embedding arithmetic |
| `httpx` | `>=0.27` | Async HTTP client for Ollama |
| `pyyaml` | `>=6.0` | YAML front-matter parsing |
| `tree-sitter` | `>=0.22` | AST parsing |
| `gitpython` | `>=3.1` | Repo cloning, git diff |
| `scipy` | `>=1.13` | Agglomerative clustering (INIT) |
| `scikit-learn` | `>=1.5` | k-means clustering (Mitosis) |
| `watchdog` | `>=4.0` | OS file system event monitoring |
| `unidiff` | `>=0.7` | Unified diff parsing |
| `click` | `>=8.1` | CLI framework |
| `rich` | `>=13.0` | CLI progress and status tables |
| `mcp` | `>=1.0` | Anthropic MCP Python SDK |
| `structlog` | `>=24.0` | Structured JSON logging |

---

## 11. Routing Architecture: Flat Retrieval Decision Record

> **Decision: Flat cosine retrieval over all buckets. Hierarchical (super-bucket) retrieval is explicitly rejected at current scale.**

### Why Hierarchical Was Considered

With 40–50 granular buckets, a concern arises that `top_k=3` cosine search might retrieve superficially similar but semantically irrelevant buckets ("retrieval noise").

A two-tier hierarchy was proposed: 5–8 "super-buckets" (architectural domains) acting as a coarse filter, followed by a fine-grained search within the winning super-bucket.

### Why Flat Wins

**Gating failure.** Hierarchical routing has a catastrophic failure mode with no recovery: if the query's embedding lands in the wrong super-bucket, the correct granular bucket is never queried. Flat search always scans all candidates; a noisy centroid returns a slightly wrong bucket but top-3 usually still includes the correct one.

**Scale doesn't justify it.** O(N) dot products over 50 unit vectors takes <1 ms. Hierarchy buys O(√N) speedup — irrelevant until N > 2000 (multi-service monorepo territory).

**Double maintenance burden.** Super-bucket centroids must be recomputed whenever any child bucket's centroid drifts after an UPDATE event. This doubles registry complexity with no query quality benefit at small N.

**Title-boosted centroids already provide semantic anchoring.** `centroid = normalize(0.8*chunk_mean + 0.2*embed(domain_label))` embeds the bucket's semantic identity directly into its routing vector. This is the key benefit super-buckets were intended to provide — without a second tier.

### Trigger for Reconsideration

Measure the false-positive retrieval rate (queries returning ≥1 irrelevant bucket in top-3) empirically. If it exceeds **20% at N > 500 buckets**, revisit hierarchy with measured data rather than theoretical concern.
