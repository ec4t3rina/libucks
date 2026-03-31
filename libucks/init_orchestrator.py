"""InitOrchestrator — walks a local repo, parses, embeds, clusters, and seeds buckets."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import struct
from pathlib import Path
from typing import List, Tuple

import numpy as np
import structlog

from libucks.config import Config
from libucks.embeddings.embedding_service import EmbeddingService
from libucks.models.chunk import ChunkMetadata, RawChunk
from libucks.parsing.ast_parser import ASTParser
from libucks.parsing.grammar_registry import SUPPORTED_LANGUAGES
from libucks.storage.bucket_registry import BucketRegistry
from libucks.storage.bucket_store import BucketStore

log = structlog.get_logger(__name__)

# Extensions considered source code (subset of grammar registry support).
_SOURCE_EXTENSIONS = set(SUPPORTED_LANGUAGES.keys())

# Approximate tokens per word for rough counting.
_TOKENS_PER_CHAR = 0.25


def _rough_tokens(text: str) -> int:
    return max(1, int(len(text) * _TOKENS_PER_CHAR))


def _encode_centroid(arr: np.ndarray) -> str:
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode()


def _chunk_id(raw: RawChunk) -> str:
    key = f"{raw.source_file}:{raw.start_line}:{raw.end_line}"
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def _to_chunk_metadata(raw: RawChunk) -> ChunkMetadata:
    return ChunkMetadata(
        chunk_id=_chunk_id(raw),
        source_file=raw.source_file,
        start_line=raw.start_line,
        end_line=raw.end_line,
        git_sha="init",
        token_count=_rough_tokens(raw.content),
    )


def _collect_source_files(repo_path: Path) -> List[Path]:
    files: List[Path] = []
    for p in sorted(repo_path.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in _SOURCE_EXTENSIONS:
            continue
        # Skip hidden dirs and common non-source trees.
        parts = p.relative_to(repo_path).parts
        if any(part.startswith(".") or part in ("__pycache__", "node_modules", ".venv", "venv") for part in parts):
            continue
        files.append(p)
    return files


def _domain_label(raw_chunks: List[RawChunk]) -> str:
    """Derive a human-readable label from the file paths in a cluster."""
    paths = sorted({Path(c.source_file).stem for c in raw_chunks})
    return ", ".join(paths[:4]) or "general"


def _cluster_chunks(
    embeddings: np.ndarray,
    raw_chunks: List[RawChunk],
    total_tokens: int,
    target_tokens: int = 20_000,
) -> List[List[int]]:
    """Return lists of chunk indices, one list per cluster."""
    n = len(raw_chunks)
    n_clusters = max(1, min(total_tokens // target_tokens, n))

    if n_clusters == 1 or n <= n_clusters:
        return [list(range(n))]

    from scipy.cluster.hierarchy import fcluster, linkage  # type: ignore[import-untyped]

    Z = linkage(embeddings, method="ward", metric="euclidean")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    groups: dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        groups.setdefault(int(label), []).append(idx)
    return list(groups.values())


class InitOrchestrator:
    def __init__(self, repo_path: Path) -> None:
        self._repo = repo_path.resolve()
        self._cfg = Config.load(self._repo)
        bucket_dir = self._repo / self._cfg.paths.bucket_dir
        registry_path = self._repo / self._cfg.paths.registry_file
        self._store = BucketStore(bucket_dir)
        self._registry = BucketRegistry(registry_path)
        self._parser = ASTParser()
        self._embedder = EmbeddingService.get_instance(self._cfg.model.embedding_model)

    async def run(self) -> None:
        log.info("init.start", repo=str(self._repo))

        files = _collect_source_files(self._repo)
        log.info("init.files_found", count=len(files))

        all_raw: List[RawChunk] = []
        for f in files:
            try:
                chunks = self._parser.parse_file(f)
                all_raw.extend(chunks)
            except Exception as exc:
                log.warning("init.parse_error", file=str(f), error=str(exc))

        if not all_raw:
            log.warning("init.no_chunks_extracted")
            return

        log.info("init.chunks_extracted", count=len(all_raw))

        contents = [c.content for c in all_raw]
        embeddings = self._embedder.embed_batch(contents)  # (N, D)

        total_tokens = sum(_rough_tokens(c.content) for c in all_raw)
        log.info("init.total_tokens", total=total_tokens)

        cluster_groups = _cluster_chunks(
            embeddings, all_raw, total_tokens, self._cfg.routing.mitosis_threshold
        )
        log.info("init.clusters", count=len(cluster_groups))

        for i, indices in enumerate(cluster_groups):
            cluster_chunks = [all_raw[j] for j in indices]
            cluster_embeddings = embeddings[indices]

            centroid = np.mean(cluster_embeddings, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid /= norm

            bucket_id = hashlib.sha1(
                "".join(c.source_file for c in cluster_chunks).encode()
            ).hexdigest()[:8]

            domain = _domain_label(cluster_chunks)
            chunk_metas = [_to_chunk_metadata(r) for r in cluster_chunks]
            prose = f"# {domain}\n\nInitial seed — {len(chunk_metas)} chunks from: {domain}.\n"

            self._store.create(
                bucket_id=bucket_id,
                domain_label=domain,
                centroid=_encode_centroid(centroid),
                chunks=chunk_metas,
                prose=prose,
            )
            await self._registry.register(
                bucket_id, centroid.astype(np.float32), sum(c.token_count for c in chunk_metas)
            )
            log.info("init.bucket_created", bucket_id=bucket_id, domain=domain, chunks=len(indices))

        self._registry.save()
        log.info("init.complete", buckets=len(cluster_groups))
