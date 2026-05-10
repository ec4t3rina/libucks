# libucks — Librarian Buckets

**Local AI Memory Server for Coding Agents**

---

## The Brutal Reality

Coding agents operating on large repositories are broken by design. Three compounding failure modes kill them in production:

1. **Context bloat.** 100,000+ line repos do not fit in a context window. Agents that try to read everything either truncate silently or blow the token budget on the first query.
2. **"Lost in the middle" degradation.** LLM attention is not uniform. Content buried in the middle of a 128K context window gets ~40% less weight than content at the boundaries. Your critical auth logic, sitting at position 60K, is functionally invisible.
3. **Runaway API cost.** Re-reading unchanged files on every query is burning money on redundant computation. A file that hasn't changed since yesterday doesn't need to be re-read today.

Every existing solution — RAG pipelines, context compression, file summarizers — patches one failure mode while ignoring the others. They're also stateless: no memory between queries, no awareness of code evolution, no understanding of domain boundaries.

`libucks` is not a patch. It is a persistent, structured memory layer — a swarm of domain-specific context buckets that sit between the coding agent and the repository, update asynchronously as code changes, and serve compressed, semantically-routed context to the agent via the Model Context Protocol (MCP). **The agent never reads raw files. It queries memory.**

---

## The Architecture

```
CODING AGENT (Claude Code)
        │  libucks_query("how does auth work?")
        ▼
    MCP BRIDGE          ← stdio transport, versioned tool schemas
        │
    TRANSLATOR          ← ONLY component that outputs natural language
        │
  CENTRAL AGENT         ← embedding-based cosine router
   /    |    \
Lib A  Lib B  Lib C     ← per-bucket async event loops
        |
   BUCKET STORE         ← .libucks/buckets/*.md with YAML front-matter
        ▲
   WATCHDOG             ← OS file events + git diff, zero AI inference
        ▲
TARGET REPOSITORY
```

**Buckets** are Markdown files with embedded YAML front-matter. Each one is a condensed, domain-specific slice of context — authentication middleware, CLI command registration, database schema, etc. — written by an autonomous Librarian agent and indexed by a centroid embedding:

```yaml
bucket_id: "a3f8c2d1"
domain_label: "authentication middleware"
centroid_embedding: "<base64 float32 array>"
token_count: 1842
chunks:
  - chunk_id: "c001"
    source_file: "src/auth/middleware.py"
    start_line: 12
    end_line: 47
    git_sha: "e4f9a3b"
```

**Routing** is flat cosine similarity over all bucket centroids — O(N) dot products over unit vectors. Sub-millisecond for up to 2,000 buckets. No hierarchical index, no gating failure modes. Title-boosted centroids (`0.8 × chunk_mean + 0.2 × embed(domain_label)`) anchor each bucket's vector identity to its semantic purpose, not just its surface text.

**The Watchdog** is a pure-Python process with zero AI inference. It listens for OS-level `FileModifiedEvent` callbacks, runs `git diff HEAD -- <file> --find-renames`, parses the unified diff into structured `DiffHunk` objects, and drops them onto the Central Agent's async queue. Rename events are detected and converted directly — no ghost context from delete+create pairs. It never stalls, never OOMs, and can be restarted independently of every other component.

---

## The Black Magic

### V2 Latent Space Communication — Bypassing English Entirely

V1 has a fundamental information bottleneck. Every Librarian-to-Translator exchange is a round-trip through English text:

```
Librarian → reason(query, context) → English string → Translator → synthesize → English string
```

A hidden state in a 3B model carries ~40,000 bits of information per position. A token carries ~15 bits. Every English-text round-trip throws away 99.96% of the model's internal representation.

**V2 eliminates the intermediate English encoding.**

Librarians now return raw `torch.Tensor` hidden states from the last layer of `Qwen2.5-3B-Instruct`. The Translator is the only component that ever decodes — and it decodes into a LoRA-finetuned `Qwen2.5-3B-Base` receiver, not back through the Instruct model.

**Why Base, not Instruct, for the receiver?**

The Instruct model was RLHF'd on ChatML templates. Injecting arbitrary continuous vectors into it causes the model to "repair" a perceived format corruption — it treats the off-manifold latent as a broken chat turn and hallucinates structure. The Base model has no such conditioning. LoRA fine-tuning (applied to `q_proj` and `v_proj` only, rank 4) teaches it to read framed latent injections as meaningful input.

**The injection protocol:**

```
inputs_embeds = [e(<bop>), h_1, ..., h_K, e(<eop>), query_tokens]
```

`<bop>` and `<eop>` are recycled from Qwen's native `<|im_start|>` / `<|im_end|>` boundary tokens — no vocabulary resize, no new embeddings, no changes to `embed_tokens` or `lm_head`. The frame is structurally identical to what the model has already processed billions of times. The LoRA delta teaches it what to do when latents appear inside the frame.

**The training objective:**

```
L_total = L_task - λ_sep · L_sep

L_task = CrossEntropy(generated_tokens | injected_framed_latents)   # teacher forcing
L_sep  = JSD(logits_correct_latent, logits_wrong_latent)            # separation signal
```

`L_task` is the only real decodability signal — cross-entropy under teacher forcing, where the teacher is `Qwen2.5-3B-Instruct` generating English summaries via the Anthropic API. Minimizing `-L_sep` (maximizing Jensen-Shannon divergence between correct and mismatched latent distributions) prevents the receiver from ignoring the latent prefix entirely and collapsing to unconditional language modeling.

**Curriculum mixing** bridges the token manifold and the latent manifold during training:

```
H^(r) = [e_1, ..., e_{⌊r·K⌋}] ⊕ [h_{⌊r·K⌋+1}, ..., h_K]
          ← token embeddings →      ←      latents       →
r ~ U[0, 1] at each step
```

`r = 0` is pure latent. `r = 1` is pure token embeddings. Uniform sampling forces the receiver to handle any mixture. Ablations: removing curriculum drops decode success from 70% to 33%.

**The CommunicationAdapter** aggregates N variable-length Librarian tensors into a fixed-length `(K=32, D)` soft-prompt before injection:

1. **Attentive Pooling** — a learned query vector cross-attends over each Librarian's token positions → one `(D,)` summary per Librarian.
2. **Inter-Librarian Self-Attention** — 2-layer, 8-head self-attention over the N summaries, capturing cross-bucket relationships.
3. **Output Projection** — K learned output queries cross-attend over refined summaries → `(32, D)` soft-prompt.

~2M trainable parameters. The backbone is always frozen.

---

### AST-Parsed Module-Affinity Agglomerative Clustering

INIT doesn't use naive k-means on raw embeddings. It uses a **module-affinity distance matrix** built from structural code relationships, fed into scipy's agglomerative hierarchical clustering.

The affinity score between any two chunks:

```
affinity(i, j) = clip(
    cosine_sim(embed_i, embed_j)
    + 0.4  if same_source_file
    + 0.2  if file_A_imports_file_B_stem (or vice versa),
    0, 1
)
```

Import detection uses `ast.parse` + `ast.walk` on the raw chunk content — no subprocess, no language server, no tree-sitter round-trip. The stem-level check (`Path(f).stem in imported_modules`) catches relative imports, aliased imports, and `from X import Y` patterns simultaneously.

This keeps logically coupled code in the same bucket even when surface-text embeddings diverge. A `middleware.py` chunk that imports `jwt_utils` stays co-located with the `jwt_utils.py` chunks even if they describe syntactically different operations. Pure embeddings would split them.

**`ContextCondenser`** — a second pure-Python, zero-inference component — produces a token-budget-safe digest of source code for the INIT prose-generation call. Priority order: module-level docstring → function/class signatures + docstrings → body lines → hard truncation. The encoder (`all-MiniLM-L6-v2`) has a 256-token hard limit; the condenser guarantees the digest fits within 200 tokens. No torch, no model, no subprocess — just `ast.parse` and `ast.unparse`.

---

## The Setup

```bash
pip install libucks
# or with V2 latent support:
pip install "libucks[latent]"
```

```bash
# Index a repository
libucks init --local /path/to/your/repo

# Train the Communication Adapter (Phase 1: MSE alignment)
libucks train-adapter --repo /path/to/your/repo

# Train the LoRA Base receiver (Phase 2: Interlat-Lite)
libucks train-adapter --repo /path/to/your/repo --train-receiver

# Start the MCP server
libucks serve
```

Add to your Claude Code config:

```json
{
  "mcpServers": {
    "libucks": {
      "command": "libucks",
      "args": ["serve"]
    }
  }
}
```

Your agent now has a single tool: `libucks_query(query, top_k=3)`. It never reads raw files again.

---

**Key dependencies:** `sentence-transformers`, `tree-sitter`, `watchdog`, `scipy`, `mcp`, `torch` (optional, V2 only), `transformers` (optional, V2 only).

**V2 hardware targets:** Apple Silicon (MPS, float16), CUDA, CPU fallback. Qwen2.5-3B-Instruct fits in ~6 GB at float16, ~2 GB at 4-bit NF4.

**Strategy is switchable at config time.** `strategy = "text"` runs V1 with async Anthropic API calls. `strategy = "latent"` activates the full V2 pipeline. Zero changes to QueryOrchestrator, CentralAgent, BucketStore, or BucketRegistry.

---

*`libucks` — because your agent shouldn't be reading the same file for the hundredth time.*
