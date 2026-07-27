"""MCP Bridge — exposes libucks tools over the Model Context Protocol (stdio).

Tools:
  libucks_query(query, top_k=3)  — query the memory store, returns synthesized answer
  libucks_status()               — bucket count and token totals
  (LoRA loaded with r=16 to match lsep_v3 training — see _cli.py:691)
"""
from __future__ import annotations

# Must be set before any native extension (tokenizers Rust runtime, ObjC) is
# imported — placing them here, at module load, guarantees that.
import os
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")  # prevents SIGABRT on Apple Silicon
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")             # prevents HF tokenizer deadlock warning
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")             # suppresses "BertModel report" etc.
os.environ.setdefault("TQDM_DISABLE", "1")                           # suppresses "Loading weights" progress bars
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")    # disables MPS pool pre-reservation; does not
                                                                      # fix peak single-op allocation (see model_manager.py
                                                                      # attn_implementation="eager") but reduces background
                                                                      # fragmentation pressure on unified memory

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server

from libucks.background_tasks import spawn
from libucks.config import Config
from libucks.storage.bucket_registry import BucketRegistry
from libucks.storage.bucket_store import BucketStore


def _load_repo_path() -> Path:
    """Return the repository root.

    Resolution order:
    1. LIBUCKS_REPO_PATH env var — explicit override for CI/scripting.
    2. ~/.libucks/active_repo — written by `libucks use <path>`.
    3. Project root inferred from __file__ — reliable fallback that is never
       the filesystem root, even when Claude Desktop launches with cwd='/'.
    """
    env_path = os.environ.get("LIBUCKS_REPO_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()
    active_repo_file = Path.home() / ".libucks" / "active_repo"
    if active_repo_file.exists():
        stored = active_repo_file.read_text().strip()
        if stored:
            return Path(stored).resolve()
    # __file__ = libucks/mcp_bridge.py → .parent = libucks/ → .parent = project root
    return Path(__file__).parent.parent.resolve()


async def serve() -> None:
    # Route ALL logging to stderr — stdout is reserved for MCP JSON-RPC.
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    import structlog
    structlog.configure(
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )

    try:
        import transformers
        transformers.logging.set_verbosity_error()
    except Exception:
        pass

    repo_path = _load_repo_path()
    cfg = Config.load(repo_path)
    registry_path = repo_path / cfg.paths.registry_file
    bucket_dir = repo_path / ".libucks"
    bucket_store_dir = repo_path / cfg.paths.bucket_dir
    print(f"[libucks] repo={repo_path}  registry={registry_path}  buckets={bucket_store_dir}", file=sys.stderr)

    # Lightweight setup — no model loading yet.
    registry = BucketRegistry(registry_path)
    registry.load()
    store = BucketStore(bucket_store_dir)

    # _ready gates call_tool until background loading finishes.
    # _state holds the objects produced by _load_heavy().
    # _load_error holds the exception message if loading fails.
    _ready: asyncio.Event = asyncio.Event()
    _state: dict[str, Any] = {}
    _load_error: list[str] = []

    async def _load_heavy() -> None:
        # Redirect stdout → stderr so stray HF prints never reach the MCP pipe.
        # This is safe to do inside the background task because mcp.server.stdio
        # has already captured the real stdout by the time this runs.
        _real_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            loop = asyncio.get_event_loop()

            # All imports and synchronous model loading run in a thread executor so
            # the event loop stays responsive for MCP handshake messages (initialize,
            # list_tools) while the ~25s setup runs in the background.
            def _sync_setup():
                from libucks.central_agent import CentralAgent
                from libucks.embeddings.embedding_service import EmbeddingService
                from libucks.librarian import Librarian
                from libucks.thinking import create_strategy
                from libucks.translator import Translator

                embedder = EmbeddingService.get_instance(cfg.model.embedding_model)
                strategy = create_strategy(cfg)

                if cfg.model.strategy == "latent":
                    strategy._mgr.load_base_model(
                        model_id=cfg.model.base_model,
                        quantization=cfg.model.quantization,
                        bnb_4bit_compute_dtype=cfg.model.bnb_4bit_compute_dtype,
                        device=cfg.model.device,
                        base_model_dtype=cfg.model.base_model_dtype,
                    )
                    print(f"[libucks] base receiver loaded: {cfg.model.base_model}", file=sys.stderr)

                    lora_path = bucket_dir / "lora_receiver.pt"
                    if lora_path.exists():
                        import torch
                        from libucks.thinking.model_manager import ModelManager as _MM
                        from libucks.thinking.training.lora_trainer import _inject_lora, _LORA_TARGETS
                        _resolved = _MM._resolve_device(cfg.model.device)
                        _inject_lora(strategy._mgr.get_base_model(), _LORA_TARGETS, r=16, alpha=16.0)
                        _lora_state = torch.load(lora_path, map_location=_resolved, weights_only=True)
                        strategy._mgr.get_base_model().load_state_dict(_lora_state, strict=False)
                        print(f"[libucks] LoRA receiver weights loaded from {lora_path}", file=sys.stderr)

                from libucks.chunk_retriever import ChunkRetriever
                chunk_retriever = ChunkRetriever(
                    cache_dir=bucket_dir / "chunk_emb_cache",
                    embedder=embedder,
                    store=store,
                )
                agent = CentralAgent(
                    registry, cfg, embed_fn=embedder.embed,
                    chunk_retriever=chunk_retriever,
                )

                # Build the CommunicationAdapter + Translator BEFORE Librarians
                # so each Librarian can be given a Translator reference. Without
                # it, Librarian._handle_update falls through to the
                # `updated_prose = current_prose` branch and prose never updates
                # on commit — silently breaking the self-evolving demo loop.
                adapter = None
                if cfg.model.strategy == "latent":
                    from libucks.thinking.communication_adapter import CommunicationAdapter
                    from libucks.thinking.model_manager import ModelManager as _MM
                    from transformers import AutoConfig as _AC
                    _adapter_device = _MM._resolve_device(cfg.model.device)
                    _enc_dim  = _AC.from_pretrained(cfg.model.local_model).hidden_size
                    _base_dim = _AC.from_pretrained(cfg.model.base_model).hidden_size
                    adapter = CommunicationAdapter(hidden_dim=_enc_dim, output_dim=_base_dim,
                                                   output_len=cfg.model.output_len)
                    adapter.load_saved_weights(bucket_dir / "adapter.pt")
                    _adapter_dtype = strategy._mgr.get_base_model().dtype
                    adapter = adapter.to(device=_adapter_device, dtype=_adapter_dtype)

                translator = Translator(
                    strategy, adapter=adapter, store=store,
                    hybrid=cfg.model.hybrid_retrieval,
                    verbatim_max_chars=cfg.model.hybrid_verbatim_max_chars,
                    chunk_retriever=chunk_retriever,
                )

                librarians: dict[str, Librarian] = {}
                for bucket_id in registry.get_all_centroids():
                    lib = Librarian(
                        bucket_id=bucket_id,
                        store=store,
                        registry=registry,
                        strategy=strategy,
                        embedder=embedder,
                        mitosis_threshold=cfg.routing.mitosis_threshold,
                        repo_path=repo_path,
                        translator=translator,
                        chunk_retriever=chunk_retriever,
                    )
                    librarians[bucket_id] = lib
                    agent.register_librarian(bucket_id, lib)

                return (embedder, strategy, agent, librarians, adapter,
                        translator, chunk_retriever)

            (embedder, strategy, agent, librarians, adapter, translator,
             chunk_retriever) = await loop.run_in_executor(None, _sync_setup)

            # ------------------------------------------------------------------
            # Startup recovery: replay commits that arrived while server was offline.
            # ------------------------------------------------------------------
            from libucks.diff.diff_extractor import DiffExtractor
            from libucks.git_hook_receiver import serve_socket
            from libucks.health_monitor import HealthMonitor
            from libucks.merging_service import MergingService
            from libucks.mitosis import MitosisService
            from libucks.novel_bucket_service import NovelBucketService
            from libucks.query_orchestrator import QueryOrchestrator
            from libucks.stale_checker import StaleChecker
            from libucks.startup_recovery import StartupRecovery

            recovery: StartupRecovery | None = None
            try:
                extractor = DiffExtractor(repo_path)
                recovery = StartupRecovery(
                    repo_path=repo_path,
                    registry=registry,
                    store=store,
                    librarians=librarians,
                    extractor=extractor,
                    central_agent=agent,
                    embedder=embedder,
                    min_bucket_seed_tokens=cfg.routing.min_bucket_seed_tokens,
                )
                current_head = await recovery.run()
                if current_head is not None:
                    registry._meta["last_indexed_head"] = current_head
                    registry._meta["watcher_pid"] = os.getpid()
                    registry.save()
                    print(f"[libucks] startup recovery complete, HEAD={current_head[:8]}", file=sys.stderr)
            except Exception as exc:
                # Recovery is best-effort — never block server startup.
                print(f"[libucks] startup recovery skipped: {exc}", file=sys.stderr)

            # ------------------------------------------------------------------
            # Git hook socket listener (Phase 6-D).
            #
            # Update architecture: git post-commit hook → Unix socket → recovery.run()
            # WatchdogService (OS file events) is intentionally NOT started here.
            # Updates are driven by git commit boundaries, not every save. This keeps
            # bucket state consistent with what is actually committed to the repo.
            # WatchdogService is available as an opt-in for non-git directories but is
            # not appropriate as the default — triggering on every file write would
            # cause partial/uncommitted state to pollute the index.
            # ------------------------------------------------------------------
            sock_path = bucket_dir / "server.sock"

            async def _on_hook_event(payload: dict) -> None:
                if recovery is None:
                    return
                try:
                    new_head = await recovery.run()
                    if new_head is not None:
                        registry._meta["last_indexed_head"] = new_head
                        registry.save()
                        print(f"[libucks] hook event '{payload.get('event')}' → re-indexed HEAD={new_head[:8]}", file=sys.stderr)
                except Exception as exc:
                    print(f"[libucks] hook event error: {exc}", file=sys.stderr)

            spawn(serve_socket(sock_path, _on_hook_event), name="git_hook.socket_server")

            # ------------------------------------------------------------------
            # HealthMonitor (Phase 6-E/6-F): autonomous quality guardian.
            # ------------------------------------------------------------------
            mitosis_svc = MitosisService(
                store=store,
                registry=registry,
                embedder=embedder,
                agent=agent,
                strategy=strategy,
                mitosis_threshold=cfg.routing.mitosis_threshold,
            )
            merging_svc = MergingService(
                registry=registry,
                store=store,
                agent=agent,
                embedder=embedder,
                strategy=strategy,
                # Keeps the merge limit strictly below the split threshold, so
                # merge and mitosis cannot fight each other every 5 minutes.
                mitosis_threshold=cfg.routing.mitosis_threshold,
            )
            health_monitor = HealthMonitor(
                registry=registry,
                store=store,
                mitosis_service=mitosis_svc,
                merging_service=merging_svc,
                embedder=embedder,
                mitosis_threshold=cfg.routing.mitosis_threshold,
                # Coherence runs over every bucket every `interval` seconds
                # forever. Without this the pass re-embeds every chunk in the
                # repo each tick — measured at a full performance core on a
                # 159-bucket repo. ChunkRetriever caches by (chunk_id, git_sha).
                chunk_retriever=chunk_retriever,
            )
            spawn(health_monitor.run(), name="health_monitor.run")

            # ------------------------------------------------------------------
            # NovelBucketService: drains CentralAgent.create_bucket_queue and
            # spawns new buckets when StartupRecovery (or WatchdogService, if
            # opted in) detects substantial novel content. Subjects emerge as
            # new buckets here; HealthMonitor splits existing ones from inside.
            # ------------------------------------------------------------------
            novel_bucket_svc = NovelBucketService(
                store=store,
                registry=registry,
                embedder=embedder,
                agent=agent,
                strategy=strategy,
                translator=translator,
                mitosis_threshold=cfg.routing.mitosis_threshold,
                repo_path=repo_path,
            )
            spawn(novel_bucket_svc.run(), name="novel_bucket_service.run")

            # ------------------------------------------------------------------
            # StaleChecker + reindex callback (Phase 6-C JIT invalidation).
            # ------------------------------------------------------------------
            stale_checker = StaleChecker(registry=registry, store=store, repo_path=repo_path)

            async def _reindex_stale(stale_bucket_ids: list[str]) -> None:
                if recovery is None:
                    return
                try:
                    new_head = await recovery.run()
                    if new_head is not None:
                        registry._meta["last_indexed_head"] = new_head
                        registry.save()
                except Exception as exc:
                    print(f"[libucks] background reindex error: {exc}", file=sys.stderr)

            orchestrator = QueryOrchestrator(
                central_agent=agent,
                librarians=librarians,
                embed_fn=embedder.embed,
                top_k=cfg.routing.top_k,
                stale_checker=stale_checker,
                reindex_fn=_reindex_stale,
            )

            _state["orchestrator"] = orchestrator
            _state["translator"] = translator
            _state["embedder"] = embedder
            print("[libucks] ready", file=sys.stderr)

        except Exception as exc:
            _load_error.append(str(exc))
            print(f"[libucks] startup failed: {exc}", file=sys.stderr)
        finally:
            sys.stdout = _real_stdout
            _ready.set()

    server = Server("libucks")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="libucks_query",
                description="Query the libucks memory store for context about the repository.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural language question"},
                        "top_k": {"type": "integer", "description": "Number of buckets to consult", "default": 3},
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="libucks_status",
                description="Return system health: bucket count and token totals.",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
        try:
            await asyncio.wait_for(_ready.wait(), timeout=120.0)
        except asyncio.TimeoutError:
            return [types.TextContent(type="text", text=
                "libucks models are still loading (>120s). "
                "Try again in a moment — check stderr for errors if this persists.")]
        if _load_error:
            return [types.TextContent(type="text", text=f"libucks startup failed: {_load_error[0]}")]

        if name == "libucks_query":
            orchestrator = _state["orchestrator"]
            translator = _state["translator"]
            embedder = _state["embedder"]
            query_text = arguments["query"]
            top_k = int(arguments.get("top_k", cfg.routing.top_k))
            orchestrator._top_k = top_k

            print(f"[libucks] query: routing '{query_text[:60]}' (top_k={top_k})", file=sys.stderr, flush=True)
            query_embedding = embedder.embed(query_text)
            pairs = await orchestrator.query(query_text)
            bucket_ids = [bid for bid, _ in pairs]
            representations = [rep for _, rep in pairs]
            print(f"[libucks] query: got {len(representations)} representations", file=sys.stderr, flush=True)
            answer = await translator.synthesize(
                query_text, representations,
                bucket_ids=bucket_ids,
                query_embedding=query_embedding,
            )
            print(f"[libucks] query: synthesis complete ({len(answer)} chars)", file=sys.stderr, flush=True)
            return [types.TextContent(type="text", text=answer)]

        if name == "libucks_status":
            centroids = registry.get_all_centroids()
            bucket_ids = list(centroids.keys())
            total_tokens = sum(
                registry.get_token_count(bid) for bid in bucket_ids
            )
            status = {
                "bucket_count": len(bucket_ids),
                "total_tokens": total_tokens,
                "buckets": {
                    bid: {"token_count": registry.get_token_count(bid)}
                    for bid in bucket_ids
                },
            }
            import json
            return [types.TextContent(type="text", text=json.dumps(status, indent=2))]

        raise ValueError(f"Unknown tool: {name!r}")

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        spawn(_load_heavy(), name="startup.load_heavy")
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )
