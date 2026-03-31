# libucks — Implementation Plan

**Test-Driven, Phase-Gated Development**

> Every phase has a Testing Gate. The gate must pass fully before Phase N+1 begins.
> No phase is considered complete until its tests are green.

---

## Phase Overview

| Phase | Name | What We Build | Gate Metric |
|---|---|---|---|
| 1 | Foundation | Data models, storage, embedding service, Strategy interface | Unit tests: 100% pass, no Ollama required |
| 2 | Central Agent | Routing math, novelty detection, registry, event types | Routing accuracy ≥ 90% with mock embeddings |
| 3 | INIT Workflow | AST parser, grammar registry, init orchestrator, CLI | Integration test on fixture repo |
| 4 | UPDATE Workflow | Watchdog, diff extractor, librarians, mitosis | Live diff test on temp git repo |
| 5 | QUERY + MCP | Translator, MCP bridge, query orchestrator | MCP Inspector tool call returns valid response |
| 6 | Hardening | Persistence recovery, health monitor, observability | Full suite green; chaos tests pass |

---

## Phase 1 — Foundation: Data Contracts, Storage, and the Strategy Interface

### What We Build

This phase produces no AI behavior. Its sole job is establishing the data contracts that every subsequent phase depends on. Nothing in Phase 2+ is built without these primitives.

**1.1 — Data Models** (`libucks/models/`)

- `ChunkMetadata`: `chunk_id`, `source_file`, `start_line`, `end_line`, `git_sha`, `token_count`
- `BucketFrontMatter`: `bucket_id`, `domain_label`, `centroid_embedding` (base64 float32), `token_count`, `chunks: List[ChunkMetadata]`
- All Event dataclasses: `DiffEvent`, `UpdateEvent`, `TombstoneEvent`, `PathUpdateEvent`, `QueryEvent`, `CreateBucketEvent`
- All models implemented with `pydantic v2` for validation and serialization.

**1.2 — BucketStore** (`libucks/storage/bucket_store.py`)

- `create(bucket_id, domain_label, centroid, chunks, prose) -> Path`
- `read(bucket_id) -> tuple[BucketFrontMatter, str]` (front-matter + prose body)
- `write_prose(bucket_id, prose: str)` — updates only the body, preserves front-matter
- `write_front_matter(bucket_id, front_matter: BucketFrontMatter)` — updates only YAML header
- `delete(bucket_id)`
- `list_all() -> List[str]` — returns all bucket_ids on disk
- Stores files at `.libucks/buckets/<bucket_id>.md`

**1.3 — BucketRegistry** (`libucks/storage/bucket_registry.py`)

- In-memory singleton. Persisted to `.libucks/registry.json` on every write.
- Per-bucket state: `{centroid: np.ndarray, token_count: int, lock: asyncio.Lock, is_splitting: bool}`
- `register(bucket_id, centroid, token_count)`
- `deregister(bucket_id)`
- `get_all_centroids() -> Dict[str, np.ndarray]`
- `set_splitting(bucket_id, flag: bool)`
- `save()` / `load()` — JSON persistence (centroid as base64)

**1.4 — EmbeddingService** (`libucks/embeddings/embedding_service.py`)

- Singleton wrapper around `sentence-transformers`.
- `embed(text: str) -> np.ndarray` — L2-normalized output
- `embed_batch(texts: List[str]) -> np.ndarray`
- Model is configurable from `Config`; default `all-MiniLM-L6-v2`.
- Model loads once at process start. Subsequent calls reuse the loaded model.

**1.5 — ThinkingStrategy Interface** (`libucks/thinking/`)

```python
# base.py
Representation = Union[str, "torch.Tensor"]

class ThinkingStrategy(ABC):
    @abstractmethod
    async def encode(self, text: str) -> Representation: ...

    @abstractmethod
    async def reason(self, query: str, context: str) -> Representation: ...

    @abstractmethod
    async def decode(self, result: Representation) -> str: ...
```

- `TextStrategy` (V1): `encode` returns text unchanged; `reason` constructs a prompt and calls Ollama via async `httpx`; `decode` returns text unchanged. All Ollama calls are async with configurable timeout.
- `LatentStrategy` (V2 stub): all three methods raise `NotImplementedError("V2 latent space — see upgrade plan")`. Exists to ensure the interface is exercised in tests.

**1.6 — Config** (`libucks/config.py`)

- Dataclass loaded from `.libucks/config.toml` via `tomllib`.
- Provides typed access to all configuration values with documented defaults.
- `Config.load(repo_path: Path) -> Config`

**1.7 — Project scaffold**

- `pyproject.toml` with all Phase 1 dependencies and optional `[dev]` group.
- `main.py` with a `click` group stub: `libucks` command with `--version`.
- `.gitignore` entry for `.libucks/`.

---

### Phase 1 — Testing Gate

**Tool:** `pytest` with `respx` for mocking `httpx` calls. **No Ollama required.**

```
tests/unit/
├── test_models.py
├── test_bucket_store.py
├── test_bucket_registry.py
├── test_embedding_service.py
├── test_text_strategy.py
└── test_config.py
```

**test_models.py**
- Round-trip serialization: `ChunkMetadata → dict → ChunkMetadata` produces identical object.
- Round-trip serialization for all Event types.
- Pydantic validation rejects missing required fields.

**test_bucket_store.py** (uses `tmp_path` pytest fixture, no real git repo)
- `create()` writes a `.md` file with valid YAML front-matter parseable by `pyyaml`.
- `read()` on a newly created bucket returns the correct `BucketFrontMatter` and prose string.
- `write_prose()` updates body; YAML front-matter is preserved byte-for-byte.
- `write_front_matter()` updates YAML; prose body is preserved.
- `delete()` removes the file; subsequent `read()` raises `FileNotFoundError`.
- `list_all()` returns exactly the created bucket IDs.

**test_bucket_registry.py**
- `register()` then `get_all_centroids()` returns the registered centroid.
- `save()` + `load()` round-trip preserves centroid within float32 tolerance (`np.allclose`).
- `set_splitting(bucket_id, True)` is reflected in subsequent reads.
- Concurrent `asyncio` writes to different buckets do not cause data corruption (run 100 concurrent `register` calls with `asyncio.gather`).

**test_embedding_service.py**
- Output shape is `(384,)` for default model.
- Output is L2-normalized: `np.linalg.norm(embed(text))` is within `1e-6` of 1.0.
- `embed_batch(["a", "b"])` shape is `(2, 384)`.
- Identical inputs produce identical outputs (determinism check).

**test_text_strategy.py** (uses `respx` to mock Ollama at `http://localhost:11434`)
- `encode(text)` returns the input text unchanged.
- `reason(query, context)` sends a POST to `/api/generate`, returns the mocked response string.
- `decode(result)` returns the input string unchanged.
- `LatentStrategy.reason(...)` raises `NotImplementedError`.

**Gate:** `pytest tests/unit/ -v` → 100% pass, 0 failures, 0 errors.

---

## Phase 2 — Central Agent: Routing, Novelty Detection, and the Registry Loop

### What We Build

**2.1 — Routing math** inside `CentralAgent`

- `route(query_embedding: np.ndarray, top_k: int) -> List[str]` — returns bucket_ids sorted by cosine similarity descending.
- `is_novel(query_embedding: np.ndarray) -> bool` — returns True if top-1 similarity < `(1 − novelty_threshold)`.
- Cosine similarity: `dot(q, c_b)` — valid because both vectors are L2-normalized.

**2.2 — CentralAgent event loop** (`libucks/central_agent.py`)

- `asyncio.Queue[DiffEvent]` for incoming Watchdog events.
- Separate `asyncio.Queue[QueryEvent]` for incoming query requests (non-blocking with respect to updates).
- UPDATE path: embed added lines → route → dispatch `UpdateEvent` to Librarian queues. If novel → `CreateBucketEvent`. If deleted lines → `TombstoneEvent`. If rename → `PathUpdateEvent`.
- Mitosis guard: if `BucketRegistry.is_splitting(bucket_id)` → put event in retry buffer, retry up to 3×.

**2.3 — Librarian (stub)** (`libucks/librarian.py`)

- Minimal implementation: receives events from its queue, logs them with `structlog`, does not yet call `ThinkingStrategy`. This is the "wiring" version — full implementation is Phase 4.

**2.4 — Mock DiffEvent fixtures** (`tests/fixtures/routing/`)

- `needle_cases.json`: 20 domains, each with a pre-computed centroid (stored as a list of floats), 5 "correct" probe queries, and 5 "adversarial" queries that are topically adjacent.
- Pre-computed at fixture generation time using real `EmbeddingService` calls. Stored as static JSON so tests are deterministic and require no embedding model at runtime.

---

### Phase 2 — Testing Gate

**Tool:** `pytest` + `pytest-asyncio`. Embeddings loaded from fixtures — no live models required.

```
tests/unit/
├── test_central_agent_routing.py
└── test_central_agent_events.py

tests/integration/
└── test_routing_accuracy.py
```

**test_central_agent_routing.py**
- `route(embed_A, top_k=3)` where `embed_A` is close to bucket_A's centroid → bucket_A is rank 1.
- `route()` with an embedding equidistant from all centroids → returns top_k results (no error).
- `is_novel()` returns True when embedding is far from all centroids (distance > threshold).
- `is_novel()` returns False when embedding is close to a centroid.

**test_central_agent_events.py** (uses `pytest-asyncio`)
- Posting a `DiffEvent` to the queue causes the correct Librarian stub to receive an `UpdateEvent` within 100ms.
- Posting a deletion hunk causes a `TombstoneEvent` (not an `UpdateEvent`).
- Posting a rename event causes a `PathUpdateEvent` with correct old and new paths.
- Posting an event while `is_splitting=True` on the target bucket: event is NOT immediately delivered; after `is_splitting` is cleared, event IS delivered.
- Novel diff (no close centroid) → `CreateBucketEvent` is emitted.

**test_routing_accuracy.py** — the Needle in a Haystack test
- Load `needle_cases.json`. For each of 20 buckets × 5 probe queries:
  - `@pytest.mark.parametrize` over all 100 cases.
  - Pre-load centroid embeddings into a mock `BucketRegistry`.
  - Call `CentralAgent.route(query_embedding, top_k=1)`.
  - Assert the correct bucket is rank 1.
- For each of 20 buckets × 5 adversarial queries:
  - Assert the correct bucket is within top-3.
- **Gate metrics:** top-1 accuracy ≥ 90%, top-3 accuracy ≥ 98%.
- `@pytest.mark.slow` tag — excluded from fast CI runs, included in full gate.

**Gate:** `pytest tests/unit/ tests/integration/test_routing_accuracy.py -v` → all pass, routing accuracy metrics printed as test output.

---

## Phase 3 — INIT Workflow: AST Parsing and Bucket Seeding

### What We Build

**3.1 — GrammarRegistry** (`libucks/parsing/grammar_registry.py`)

- Maps file extensions → Tree-sitter language names.
- `get_parser(extension: str) -> tree_sitter.Parser` — lazy-downloads compiled grammar `.so` from the tree-sitter GitHub releases API if not cached; caches under `~/.libucks/grammars/`.
- Raises a clear error for unsupported extensions (rather than silently skipping files).

**3.2 — ASTParser** (`libucks/parsing/ast_parser.py`)

- `parse_file(path: Path) -> List[RawChunk]`
- Extracts: module-level docstrings, function definitions (name + signature + docstring + body), class definitions (name + docstring + method signatures).
- Falls back to line-split chunking (every N lines) for unsupported file types.

**3.3 — RepoCloner** (`libucks/init_orchestrator.py`)

- Clones target repo via `gitpython` to `~/.libucks/repos/<repo-name>/`.
- If the repo is already cloned, performs `git fetch` + `git reset --hard origin/HEAD` to refresh.

**3.4 — InitOrchestrator** (`libucks/init_orchestrator.py`)

1. Clone/refresh repo.
2. Walk all source files; skip binary files, `.gitignore`d paths.
3. Parse each file → `List[RawChunk]`.
4. `EmbeddingService.embed_batch(all_chunk_contents)`.
5. Agglomerative clustering (scipy): `n_clusters = max(1, total_tokens // 2000)`.
6. For each cluster: `BucketStore.create()`, `Librarian.initialize(chunks)` (stub: writes raw chunk content, no summarization yet — that is Phase 4), `BucketRegistry.register()`.
7. Print rich progress table: files parsed, chunks extracted, buckets created.

**3.5 — CLI command**

- `libucks init <repo-url>` — runs `InitOrchestrator`.
- `libucks init --local <path>` — skips cloning, runs against an already-cloned directory.

**3.6 — Fixture repo** (`tests/fixtures/repos/sample_repo/`)

A small, fully committed Python project checked into the libucks repo for use in integration tests. Contains:
- `auth/middleware.py` — JWT validation functions
- `db/models.py` — ORM models
- `api/routes.py` — HTTP route handlers
- `ui/components.py` — frontend component stubs
- `utils/helpers.py` — utility functions
- `README.md` and a `setup.py`

~15 functions total, ~500 lines. Enough to seed 2–4 buckets.

---

### Phase 3 — Testing Gate

**Tool:** `pytest` with `tmp_path`, `monkeypatch`. Grammar downloads are mocked via `respx`. No live git clone in CI.

```
tests/unit/
├── test_ast_parser.py
└── test_grammar_registry.py

tests/integration/
└── test_init_workflow.py
```

**test_ast_parser.py**
- Given a Python source file string (inline fixture), `parse_file()` returns `RawChunk` objects where `start_line` and `end_line` are accurate.
- `chunk.content` contains the function signature and its docstring.
- `chunk.language == "python"`.
- A file with no top-level declarations returns at least one fallback chunk (the whole file).
- Parsing a file with a syntax error does not raise; returns best-effort chunks.

**test_grammar_registry.py** (mocks HTTP with `respx`)
- First call to `get_parser("py")` triggers a download request to the grammar release URL.
- Second call to `get_parser("py")` does NOT trigger a download (cache hit).
- Call with an unsupported extension raises `UnsupportedLanguageError` with the extension name in the message.

**test_init_workflow.py** (uses `tmp_path`, does NOT clone a real repo)
- `InitOrchestrator.run(local_path=sample_repo_fixture_path)` completes without error.
- `.libucks/buckets/` directory is created and contains at least 1 `.md` file.
- Each `.md` file has valid YAML front-matter parseable by `pyyaml`.
- Every source file in `sample_repo/` has at least one `chunk_id` present across all bucket front-matters (no file is silently dropped).
- No bucket's `token_count` exceeds `mitosis_threshold` at init time (seeding heuristic is working).
- `BucketRegistry.load()` after `InitOrchestrator.run()` returns the same bucket IDs as the `.md` files on disk.

**Gate:** `pytest tests/unit/test_ast_parser.py tests/unit/test_grammar_registry.py tests/integration/test_init_workflow.py -v` → 100% pass.

---

## Phase 4 — UPDATE Workflow: Watchdog, Diff Extractor, Librarians, Mitosis

### What We Build

This is the most complex phase. It wires the live-update pipeline end-to-end.

**4.1 — DiffExtractor** (`libucks/diff/diff_extractor.py`)

- `extract(filepath: Path) -> List[DiffEvent]`
- Runs `git diff HEAD -- <filepath> --find-renames` via `gitpython`.
- Parses unified diff into `List[DiffHunk]` using `unidiff`.
- Detects renames: if `git diff` reports a rename, emits `DiffEvent(is_rename=True, old_path=..., new_path=...)`.
- Handles binary files gracefully (skips with a log warning).

**4.2 — WatchdogService** (`libucks/watchdog_service.py`)

- Uses `watchdog.observers.Observer` and a custom `FileSystemEventHandler`.
- On `FileModifiedEvent` for a tracked source file: calls `DiffExtractor.extract()` → places `DiffEvent` on `CentralAgent.diff_queue`.
- Debounce: if the same file triggers events within 500ms, only the last event is processed. Prevents flooding on rapid auto-save.

**4.3 — Librarian (full implementation)** (`libucks/librarian.py`)

- `asyncio.Queue` per Librarian instance.
- `UpdateEvent` handler:
  1. Acquire per-bucket `asyncio.Lock`.
  2. Read current bucket from `BucketStore`.
  3. Call `ThinkingStrategy.reason(diff_content, current_prose)` → `Representation`.
  4. Call `ThinkingStrategy.decode(result)` only if final output needed (not for internal reason calls — Translator handles final decode).
  5. Write updated prose via `BucketStore.write_prose()`.
  6. Recompute centroid: `normalize(mean(embed_batch([c.content for c in chunks])))`.
  7. Update `BucketRegistry` centroid + token_count.
  8. Release lock.
  9. If `token_count > mitosis_threshold` → signal `MitosisService`.
- `TombstoneEvent` handler:
  1. Acquire lock.
  2. Remove stale `ChunkMetadata` entries matching `chunk_id` in `TombstoneEvent.chunk_ids`.
  3. Re-render prose (call `ThinkingStrategy.reason` asking it to rewrite the summary without the purged concepts).
  4. Write via `BucketStore`. Release lock.
- `PathUpdateEvent` handler: acquire lock → update `source_file` field in all matching `ChunkMetadata` → write front-matter → release lock (no prose re-render needed).
- `InitEvent` handler: generate initial condensed prose via `ThinkingStrategy.reason(chunks_as_text, "")` → write to `BucketStore`.
- `QueryEvent` handler: call `ThinkingStrategy.reason(query, prose)` → return `Representation` (no decode, no natural language output — that is the Translator's job).

**4.4 — MitosisService** (`libucks/mitosis.py`)

1. `set_splitting(bucket_id, True)` in `BucketRegistry`.
2. Read all `ChunkMetadata` from bucket front-matter.
3. `EmbeddingService.embed_batch([c.content for c in chunks])`.
4. k-means k=2 via `scikit-learn`.
5. `BucketStore.create()` for each child; `Librarian.initialize()` for each child.
6. `BucketRegistry.register()` for children; `BucketRegistry.deregister()` for parent.
7. `BucketStore.delete(parent_id)`.
8. `set_splitting(bucket_id, False)` — this triggers `CentralAgent` to drain the retry buffer.

**4.5 — Ghost context test (git sha validation)**

- In `TombstoneEvent` handler: before purging a chunk, compare `chunk.git_sha` to the deletion commit's sha. If `chunk.git_sha` is newer → skip purge (the chunk survived the deletion in a subsequent write).

---

### Phase 4 — Testing Gate

**Tool:** `pytest` + `pytest-asyncio` + `pytest-timeout` + `respx` (mock Ollama).

```
tests/unit/
├── test_diff_extractor.py
├── test_watchdog_service.py
├── test_librarian.py
└── test_mitosis.py

tests/integration/
├── test_update_workflow.py
└── test_ghost_context.py
```

**test_diff_extractor.py** (no real git required — inject mock `gitpython` output)
- Given synthetic unified diff string with an added function → `DiffHunk` has correct `added_lines`, `new_start`, `new_end`.
- Given unified diff with deletions → `removed_lines` are correct, `old_start`/`old_end` are correct.
- Given a rename diff (`old_path → new_path`) → `DiffEvent.is_rename=True`, `old_path` and `new_path` are populated.
- Binary file → no hunk emitted, warning logged.

**test_librarian.py** (mocks `ThinkingStrategy` and `BucketStore`)
- `UpdateEvent` → `ThinkingStrategy.reason` is called with the diff content and current prose.
- `TombstoneEvent` → the specified `chunk_id` is absent from the bucket front-matter after the handler runs.
- `PathUpdateEvent` → all `ChunkMetadata.source_file` for the old path are updated; prose is unchanged.
- `QueryEvent` → `ThinkingStrategy.reason` is called; `ThinkingStrategy.decode` is NOT called (the Librarian does not produce final natural language).

**test_mitosis.py**
- Build a bucket with 40 synthetic chunks: 20 near `centroid_A`, 20 near `centroid_B` (far apart).
- Run `MitosisService`.
- Assert: exactly 2 child buckets created; `len(A.chunks) + len(B.chunks) == 40`; parent absent from registry.
- Each child centroid is closer to its own chunks' embeddings than to the other child's.
- **Race condition test:** `asyncio.gather(MitosisService.run(bucket), *[librarian.handle(UpdateEvent) for _ in range(10)])` with `pytest-timeout` at 10s. Assert no deadlock, all 10 `UpdateEvent` objects are eventually processed (either in a child bucket or via the retry buffer).

**test_update_workflow.py** (uses `tmp_path`, creates a real git repo with `gitpython`)
1. `git init` in `tmp_path`. Write a Python file. `git commit`.
2. Run `InitOrchestrator` against `tmp_path`.
3. Edit the Python file (add a new function). Do NOT commit (Watchdog reads `git diff HEAD`).
4. Manually call `DiffExtractor.extract(modified_file)`.
5. Post resulting `DiffEvent` to `CentralAgent.diff_queue`.
6. `await asyncio.sleep(2)` — allow Librarian to process.
7. Assert: the new function name appears somewhere in the relevant bucket's prose or chunk metadata.
8. Assert: the bucket's `token_count` in `BucketRegistry` is updated.

**test_ghost_context.py**
- `git init` in `tmp_path`. Write a Python file with 3 functions. Commit. Run INIT.
- Assert: 3 chunk_ids recorded.
- Delete one function. `git commit`.
- Feed deletion diff through `DiffExtractor` → `CentralAgent`.
- `await asyncio.sleep(2)`.
- Assert: deleted chunk's `chunk_id` absent from bucket front-matter.
- Assert: deleted function name absent from bucket prose (string search).
- Assert: other 2 chunks intact.
- **Rename test:** `git mv` the file. Feed rename diff. Assert all `chunk.source_file` updated; no chunks lost.

**Gate:** `pytest tests/unit/ tests/integration/test_update_workflow.py tests/integration/test_ghost_context.py -v --timeout=30` → 100% pass.

---

## Phase 5 — QUERY Workflow: Translator and MCP Bridge

### What We Build

**5.1 — QueryOrchestrator** (`libucks/query_orchestrator.py`)

1. `embed(query)` → `q`.
2. `BucketRegistry.get_all_centroids()` → cosine similarity → top-K bucket_ids (read-only, no lock).
3. `asyncio.gather(*[librarian.handle(QueryEvent(query, bid)) for bid in top_k_ids])` → `List[Representation]`.
4. Pass `List[Representation]` + `query` to `Translator`.

**5.2 — Translator** (`libucks/translator.py`)

- `synthesize(query: str, representations: List[Representation]) -> str`
- Constructs a synthesis prompt from the query and all Representations (decoded to text via `ThinkingStrategy.decode` for V1; in V2 this is where tensors are decoded by the model's head).
- Calls `ThinkingStrategy.reason(synthesis_prompt, "")` → one final `Representation`.
- Calls `ThinkingStrategy.decode(result)` → `str`.
- **The only component permitted to call `decode` and return the result as final output.**
- Sanitization: before returning, strip any internal metadata keys (`bucket_id`, `chunk_id`, `git_sha`) that may have leaked into the Ollama response.

**5.3 — MCPBridge** (`libucks/mcp_bridge.py`)

- MCP server via `mcp` Python SDK, stdio transport.
- Tool: `libucks_query(query: str, top_k: int = 3) -> str` — calls `QueryOrchestrator` → `Translator` → returns string.
- Tool: `libucks_status() -> dict` — reads `BucketRegistry`: returns `{bucket_count, total_tokens, last_updated, pending_events}`.
- Tool schemas loaded from `tools_v1.json` at startup. Schema validated against MCP spec at load time with `jsonschema`.
- CLI: `libucks serve` starts the MCP server.

**5.4 — tools_v1.json** (root of repo)

```json
{
  "version": "1",
  "tools": [
    {
      "name": "libucks_query",
      "description": "Query the libucks memory system for condensed context about the target repository.",
      "inputSchema": {
        "type": "object",
        "properties": {
          "query": { "type": "string", "description": "Natural language question about the codebase." },
          "top_k": { "type": "integer", "default": 3, "description": "Number of buckets to consult." }
        },
        "required": ["query"]
      }
    },
    {
      "name": "libucks_status",
      "description": "Return current system health: bucket count, total tokens, last update timestamp.",
      "inputSchema": { "type": "object", "properties": {} }
    }
  ]
}
```

---

### Phase 5 — Testing Gate

**Tool:** `pytest` + `pytest-asyncio` + `respx` (mock Ollama) + MCP SDK in-process test client.

```
tests/unit/
├── test_translator.py
└── test_mcp_bridge.py

tests/integration/
├── test_query_workflow.py
└── test_mcp_schema.py
```

**test_translator.py** (mocks `ThinkingStrategy`)
- Given 3 mock `Representation` strings, `Translator.synthesize()` returns a non-empty string.
- Output does NOT contain the literal strings `"bucket_id"`, `"chunk_id"`, or `"git_sha"`.
- `ThinkingStrategy.decode` is called exactly once (at the end).
- `ThinkingStrategy.reason` is called to synthesize partial results (not just concatenate them).

**test_mcp_bridge.py** (uses MCP SDK in-process test client — no subprocess)
- `libucks_query(query="test")` returns a non-empty string within 30s (pytest-timeout).
- `libucks_status()` returns a dict with keys `bucket_count`, `total_tokens`, `last_updated`.
- `libucks_status().bucket_count` matches the number of `.md` files in `.libucks/buckets/`.
- Calling `libucks_query` while Ollama is down (mocked via `respx` to return 503): returns a structured error string, does NOT raise an unhandled exception.

**test_mcp_schema.py**
- Load `tools_v1.json`.
- Validate the JSON structure against the MCP tool schema specification using `jsonschema`.
- Assert that `libucks_query` input schema requires `"query"` and that `top_k` has a default.
- **Schema drift regression:** if `tools_v1.json` is modified without updating its version field, this test fails (catches accidental breaking changes).

**test_query_workflow.py** — End-to-end query path (mocks Ollama, uses real BucketStore)
1. `InitOrchestrator.run(sample_repo_fixture_path)` → seed buckets.
2. Post a `QueryEvent("how does authentication work?")` through `QueryOrchestrator`.
3. Assert returned string is non-empty.
4. Assert returned string contains content from the `auth/` related bucket (not random bucket content).
5. Assert `ThinkingStrategy.decode` was called by `Translator` and not by any `Librarian`.

**MCP Inspector manual gate** (required before Phase 6):
- `libucks init --local tests/fixtures/repos/sample_repo/`
- `libucks serve`
- Open Anthropic MCP Inspector → connect to `libucks` server.
- Call `libucks_query` with query `"what does the auth middleware do?"`.
- Verify response is coherent English and references content from `auth/middleware.py` in the fixture repo.
- Call `libucks_status` → verify JSON response with correct bucket count.
- Document: screenshot or log transcript committed to `tests/integration/mcp_inspector_log.txt`.

**Gate:** `pytest tests/unit/test_translator.py tests/unit/test_mcp_bridge.py tests/integration/ -v --timeout=60` → 100% pass + MCP Inspector manual gate documented.

---

## Phase 6 — Hardening: Persistence, Recovery, Observability, and Chaos

### What We Build

**6.1 — Startup recovery**

- On `libucks serve`, if `.libucks/` exists: reconstruct `BucketRegistry` from all `.md` files on disk. Resume Librarian event loops. No state is held exclusively in memory.
- `BucketRegistry.load()` is called at startup before accepting any MCP connections.

**6.2 — HealthMonitor** (`libucks/health_monitor.py`)

- Async coroutine, started alongside the MCP bridge.
- Pings Ollama (`GET /api/tags`) every 30 seconds.
- If Ollama is unhealthy: sets a `system_degraded` flag; incoming `DiffEvent` objects are written to `.libucks/pending_events.jsonl` instead of the live queue.
- On Ollama recovery: drains `pending_events.jsonl` back into the live queue.

**6.3 — Structured logging**

- `structlog` configured at startup to emit JSON to `.libucks/libucks.log`.
- Every log entry carries: `bucket_id`, `event_type`, `latency_ms`, `timestamp`.
- `CentralAgent`, `Librarian`, `Translator`, `MCPBridge` all bind their component name in the logger context.

**6.4 — CLI additions**

- `libucks status` — rich table: bucket name, token count, chunk count, last updated, centroid drift % since init.
- `libucks reset` — clears `.libucks/`, re-runs INIT against the cached clone.
- `libucks logs` — tails `.libucks/libucks.log` with `rich` formatting.

**6.5 — Graceful shutdown**

- `SIGTERM` handler: drain in-flight Librarian queues, flush `BucketRegistry.save()`, stop MCP server, exit 0.

---

### Phase 6 — Testing Gate

**Tool:** `pytest` + `pytest-asyncio` + `pytest-timeout` + chaos scenarios.

```
tests/integration/
├── test_startup_recovery.py
├── test_health_monitor.py
├── test_graceful_shutdown.py
└── test_chaos.py
```

**test_startup_recovery.py**
- Run `InitOrchestrator` → kill the process → restart with `BucketRegistry.load()`.
- Assert registry contains the same bucket IDs and centroids as before the kill.
- Assert a `libucks_query` call returns valid content immediately after recovery (no re-init needed).

**test_health_monitor.py** (mocks Ollama via `respx`)
- While Ollama returns 503: `DiffEvent` objects are written to `pending_events.jsonl`, not the live queue.
- When Ollama returns 200 again: events in `pending_events.jsonl` are drained and eventually processed.
- After drain: `pending_events.jsonl` is empty (or removed).

**test_chaos.py**
- **Rapid writes:** send 50 `DiffEvent` objects to the queue in 100ms. Assert all 50 are eventually processed (no events silently dropped). `pytest-timeout` at 30s.
- **Concurrent mitosis + query:** trigger mitosis on bucket A while simultaneously dispatching 20 `QueryEvent` objects routing to bucket A. Assert: no deadlock, all queries return valid responses (either from parent before split or from a child after split).
- **Disk full simulation:** mock `BucketStore.write_prose()` to raise `OSError`. Assert: `Librarian` logs the error, places the event in the retry buffer, does not crash.
- **Ollama timeout:** mock Ollama to hang for 60s. Assert: `TextStrategy.reason()` raises `httpx.TimeoutException` after configured timeout (default 30s), Librarian logs and moves on, does not block other events.

**Full suite gate:** `pytest tests/ -v --timeout=60 -x` → 100% pass.

**Performance gate** (optional, not blocking):
- Run `libucks init` on a 10 000-line public Python repo. Assert init completes in under 5 minutes.
- Run `libucks_query` end-to-end with Ollama live. Assert response time < 10s for top-3 routing.

---

## Dependency Summary

```toml
[project]
name = "libucks"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.0",
    "sentence-transformers>=3.0",
    "numpy>=1.26",
    "httpx>=0.27",
    "pyyaml>=6.0",
    "tree-sitter>=0.22",
    "gitpython>=3.1",
    "scipy>=1.13",
    "scikit-learn>=1.5",
    "watchdog>=4.0",
    "unidiff>=0.7",
    "click>=8.1",
    "rich>=13.0",
    "mcp>=1.0",
    "structlog>=24.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "pytest-timeout>=2.3",
    "respx>=0.21",
    "jsonschema>=4.22",
]
```

---

## Phase Gate Summary

| Phase | Gate Command | Pass Criteria |
|---|---|---|
| 1 | `pytest tests/unit/ -v` | 100% pass, 0 Ollama calls |
| 2 | `pytest tests/unit/ tests/integration/test_routing_accuracy.py -v` | 100% pass; top-1 ≥ 90%, top-3 ≥ 98% |
| 3 | `pytest tests/unit/ tests/integration/test_init_workflow.py -v` | 100% pass; fixture repo fully indexed |
| 4 | `pytest tests/unit/ tests/integration/test_update_workflow.py tests/integration/test_ghost_context.py -v --timeout=30` | 100% pass; live diff test < 2s |
| 5 | `pytest tests/ -v --timeout=60` + MCP Inspector manual gate | 100% pass; MCP Inspector transcript committed |
| 6 | `pytest tests/ -v --timeout=60 -x` | 100% pass; chaos tests pass |
