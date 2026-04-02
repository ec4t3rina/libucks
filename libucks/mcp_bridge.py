"""MCP Bridge — exposes libucks tools over the Model Context Protocol (stdio).

Tools:
  libucks_query(query, top_k=3)  — query the memory store, returns synthesized answer
  libucks_status()               — bucket count and token totals
"""
from __future__ import annotations

# Must be set before any native extension (tokenizers Rust runtime, ObjC) is
# imported — placing them here, at module load, guarantees that.
import os
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")  # prevents SIGABRT on Apple Silicon
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")             # prevents HF tokenizer deadlock warning
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")             # suppresses "BertModel report" etc.
os.environ.setdefault("TQDM_DISABLE", "1")                           # suppresses "Loading weights" progress bars

import asyncio
import logging
import sys
import tomllib
from pathlib import Path
from typing import Any

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server

from libucks.central_agent import CentralAgent
from libucks.config import Config
from libucks.diff.diff_extractor import DiffExtractor
from libucks.embeddings.embedding_service import EmbeddingService
from libucks.git_hook_receiver import serve_socket
from libucks.health_monitor import HealthMonitor
from libucks.librarian import Librarian
from libucks.merging_service import MergingService
from libucks.mitosis import MitosisService
from libucks.query_orchestrator import QueryOrchestrator
from libucks.stale_checker import StaleChecker
from libucks.startup_recovery import StartupRecovery
from libucks.storage.bucket_registry import BucketRegistry
from libucks.storage.bucket_store import BucketStore
from libucks.thinking.text_strategy import TextStrategy
from libucks.translator import Translator


def _load_repo_path() -> Path:
    """Return the repository root.

    Resolution order:
    1. LIBUCKS_REPO_PATH env var — set this in claude_desktop_config.json "env"
       to point the server at any repo you want.
    2. Project root inferred from __file__ — reliable fallback that is never
       the filesystem root, even when Claude Desktop launches with cwd='/'.
    """
    env_path = os.environ.get("LIBUCKS_REPO_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()
    # __file__ = libucks/mcp_bridge.py → .parent = libucks/ → .parent = project root
    return Path(__file__).parent.parent.resolve()


async def serve() -> None:
    # Route ALL logging to stderr — stdout is reserved for MCP JSON-RPC.
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, force=True)

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
    print(f"[libucks] repo={repo_path}  registry={registry_path}  buckets={bucket_dir}", file=sys.stderr)

    registry = BucketRegistry(registry_path)
    registry.load()

    store = BucketStore(bucket_dir)

    # Pre-load the embedding model BEFORE the MCP stdio server starts.
    # Temporarily redirect sys.stdout → sys.stderr so any remaining direct
    # prints from model loading never reach the MCP pipe.
    _real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        embedder = EmbeddingService.get_instance(cfg.model.embedding_model)
    finally:
        sys.stdout = _real_stdout  # restore BEFORE mcp.server.stdio.stdio_server() captures it
    strategy = TextStrategy.from_env(cfg.model.anthropic_model)

    agent = CentralAgent(registry, cfg, embed_fn=embedder.embed)

    librarians: dict[str, Librarian] = {}
    for bucket_id in registry.get_all_centroids():
        lib = Librarian(
            bucket_id=bucket_id,
            store=store,
            registry=registry,
            strategy=strategy,
            embedder=embedder,
            mitosis_threshold=cfg.routing.mitosis_threshold,
        )
        librarians[bucket_id] = lib
        agent.register_librarian(bucket_id, lib)

    translator = Translator(strategy)

    # ------------------------------------------------------------------
    # Startup recovery: replay commits that arrived while server was offline.
    # ------------------------------------------------------------------
    recovery: StartupRecovery | None = None
    try:
        extractor = DiffExtractor(repo_path)
        recovery = StartupRecovery(
            repo_path=repo_path,
            registry=registry,
            store=store,
            librarians=librarians,
            extractor=extractor,
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
    # ------------------------------------------------------------------
    sock_path = bucket_dir / "server.sock"

    async def _on_hook_event(payload: dict) -> None:
        """Handle a JSON event from a git hook and trigger background re-index."""
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

    asyncio.ensure_future(serve_socket(sock_path, _on_hook_event))

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
    )
    health_monitor = HealthMonitor(
        registry=registry,
        store=store,
        mitosis_service=mitosis_svc,
        merging_service=merging_svc,
        embedder=embedder,
        mitosis_threshold=cfg.routing.mitosis_threshold,
    )
    asyncio.ensure_future(health_monitor.run())

    # ------------------------------------------------------------------
    # StaleChecker + reindex callback (Phase 6-C JIT invalidation).
    # ------------------------------------------------------------------
    stale_checker = StaleChecker(registry=registry, store=store, repo_path=repo_path)

    async def _reindex_stale(stale_bucket_ids: list[str]) -> None:
        """Background re-index triggered by stale query results."""
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
        if name == "libucks_query":
            query_text = arguments["query"]
            top_k = int(arguments.get("top_k", cfg.routing.top_k))
            orchestrator._top_k = top_k

            representations = await orchestrator.query(query_text)
            answer = await translator.synthesize(query_text, representations)
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
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )
