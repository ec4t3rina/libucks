"""MCP Bridge — exposes libucks tools over the Model Context Protocol (stdio).

Tools:
  libucks_query(query, top_k=3)  — query the memory store, returns synthesized answer
  libucks_status()               — bucket count and token totals
"""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server

from libucks.central_agent import CentralAgent
from libucks.config import Config
from libucks.embeddings.embedding_service import EmbeddingService
from libucks.librarian import Librarian
from libucks.query_orchestrator import QueryOrchestrator
from libucks.storage.bucket_registry import BucketRegistry
from libucks.storage.bucket_store import BucketStore
from libucks.thinking.text_strategy import TextStrategy
from libucks.translator import Translator


def _load_repo_path() -> Path:
    """Find the target repo path from .libucks/config.toml in cwd, or use cwd."""
    cwd = Path.cwd()
    config_file = cwd / ".libucks" / "config.toml"
    if config_file.exists():
        with open(config_file, "rb") as fh:
            data = tomllib.load(fh)
        repo = data.get("paths", {}).get("repo_root", None)
        if repo:
            return Path(repo).expanduser().resolve()
    return cwd


async def serve() -> None:
    repo_path = _load_repo_path()
    cfg = Config.load(repo_path)

    registry_path = repo_path / cfg.paths.registry_file
    bucket_dir = repo_path / cfg.paths.bucket_dir

    registry = BucketRegistry(registry_path)
    registry.load()

    store = BucketStore(bucket_dir)
    embedder = EmbeddingService.get_instance(cfg.model.embedding_model)
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

    orchestrator = QueryOrchestrator(
        central_agent=agent,
        librarians=librarians,
        embed_fn=embedder.embed,
        top_k=cfg.routing.top_k,
    )
    translator = Translator(strategy)

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
