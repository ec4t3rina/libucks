"""InitOrchestrator — walks a local repo, parses, embeds, clusters, and seeds buckets."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import random
import struct
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import structlog

from libucks.config import Config
from libucks.embeddings.embedding_service import EmbeddingService
from libucks.models.chunk import ChunkMetadata, RawChunk
from libucks.parsing.aspect_mapper import AspectMapper
from libucks.parsing.ast_parser import ASTParser
from libucks.parsing.grammar_registry import SUPPORTED_LANGUAGES
from libucks.storage.bucket_registry import BucketRegistry
from libucks.storage.bucket_store import BucketStore
from libucks.thinking.base import ThinkingStrategy
from libucks.thinking.context_condenser import ContextCondenser

log = structlog.get_logger(__name__)

# Extensions considered source code (subset of grammar registry support).
_SOURCE_EXTENSIONS = set(SUPPORTED_LANGUAGES.keys())

# Additional extensions beyond the grammar registry that carry useful context.
_EXTRA_EXTENSIONS = {".md"}

# Files larger than this are almost certainly generated blobs or data dumps.
_MAX_FILE_BYTES = 500 * 1_024  # 500 KB

# Directory names that are pure noise — any path component matching one is skipped.
_NOISE_DIRS = {
    "__pycache__", "node_modules", ".venv", "venv",
    "dist", "build", "out",
    "assets", "static", "images", "media",
    "coverage", "htmlcov",
    ".claude",
    "docs",
    "scripts",
}

# Approximate tokens per character for rough counting.
_TOKENS_PER_CHAR = 0.25

# Token budget for ContextCondenser when preparing digest for LLM.
_CONDENSER_BUDGET = 200

# Blend weight for title embedding in centroid (0.2 title, 0.8 chunk mean).
_TITLE_BLEND_ALPHA = 0.2

# Max concurrent LLM calls — keeps throughput near 50 req/min without hammering the API.
_PROSE_CONCURRENCY = 10

# Retry config for 429 / transient errors.
_MAX_RETRIES = 4
_RETRY_BASE_DELAY = 2.0  # seconds; doubles each attempt + jitter


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
    allowed_exts = _SOURCE_EXTENSIONS | _EXTRA_EXTENSIONS
    files: List[Path] = []
    for p in sorted(repo_path.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in allowed_exts:
            continue
        parts = p.relative_to(repo_path).parts
        if any(part.startswith(".") or part in _NOISE_DIRS for part in parts):
            continue
        if p.stat().st_size > _MAX_FILE_BYTES:
            continue
        files.append(p)
    return files


def _domain_label(raw_chunks: List[RawChunk]) -> str:
    """Derive a human-readable fallback label from the file paths in a cluster."""
    paths = sorted({Path(c.source_file).stem for c in raw_chunks})
    return ", ".join(paths[:4]) or "general"


def _n_clusters(total_tokens: int, target_tokens: int, n_chunks: int) -> int:
    return max(1, min(total_tokens // target_tokens, n_chunks))


def _parse_title_and_prose(response: str, fallback_label: str) -> Tuple[str, str]:
    """Parse a TITLE: / --- structured LLM response.

    Expected format::

        TITLE: <5-10 word aspect title>
        ---
        <prose summary>

    Falls back to *fallback_label* as the title (and full response as prose)
    if the expected structure is not found.
    """
    lines = response.strip().splitlines()
    title = fallback_label
    prose = response.strip()

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("TITLE:"):
            title = stripped[len("TITLE:"):].strip()
            # Find the --- separator following the TITLE line.
            for j in range(i + 1, len(lines)):
                if lines[j].strip() == "---":
                    prose = "\n".join(lines[j + 1:]).strip()
                    break
            break

    return title, prose


class InitOrchestrator:
    def __init__(self, repo_path: Path, strategy: Optional[ThinkingStrategy] = None) -> None:
        self._repo = repo_path.resolve()
        self._cfg = Config.load(self._repo)
        bucket_dir = self._repo / self._cfg.paths.bucket_dir
        registry_path = self._repo / self._cfg.paths.registry_file
        self._store = BucketStore(bucket_dir)
        self._registry = BucketRegistry(registry_path)
        self._parser = ASTParser()
        self._embedder = EmbeddingService.get_instance(self._cfg.model.embedding_model)
        self._strategy = strategy
        self._condenser = ContextCondenser()
        self._aspect_mapper = AspectMapper()
        self._sem = asyncio.Semaphore(_PROSE_CONCURRENCY)

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

        n = _n_clusters(total_tokens, self._cfg.routing.init_bucket_size, len(all_raw))
        cluster_groups = self._aspect_mapper.cluster(all_raw, embeddings, n_clusters=n)
        log.info("init.clusters", count=len(cluster_groups))

        # ----------------------------------------------------------------
        # Phase 1 (sync): Compute centroid, bucket_id, fallback label, metas
        # ----------------------------------------------------------------
        cluster_data = []
        for indices in cluster_groups:
            cluster_chunks = [all_raw[j] for j in indices]
            cluster_embeddings = embeddings[np.array(indices)]

            centroid = np.mean(cluster_embeddings, axis=0).astype(np.float32)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid /= norm

            bucket_id = hashlib.sha1(
                "".join(c.source_file for c in cluster_chunks).encode()
            ).hexdigest()[:8]

            fallback_label = _domain_label(cluster_chunks)
            chunk_metas = [_to_chunk_metadata(r) for r in cluster_chunks]

            cluster_data.append({
                "cluster_chunks": cluster_chunks,
                "centroid": centroid,
                "bucket_id": bucket_id,
                "fallback_label": fallback_label,
                "chunk_metas": chunk_metas,
            })

        # ----------------------------------------------------------------
        # Phase 2: Generate prose (concurrent LLM calls if strategy present)
        # ----------------------------------------------------------------
        if self._strategy is not None:
            results = await asyncio.gather(
                *[self._generate_prose(d) for d in cluster_data]
            )
        else:
            results = [self._placeholder_prose(d) for d in cluster_data]

        # ----------------------------------------------------------------
        # Phase 3 (sync): Persist buckets
        # ----------------------------------------------------------------
        for result in results:
            self._store.create(
                bucket_id=result["bucket_id"],
                domain_label=result["domain"],
                centroid=_encode_centroid(result["centroid"]),
                chunks=result["chunk_metas"],
                prose=result["prose"],
            )
            await self._registry.register(
                result["bucket_id"],
                result["centroid"].astype(np.float32),
                sum(c.token_count for c in result["chunk_metas"]),
            )
            log.info(
                "init.bucket_created",
                bucket_id=result["bucket_id"],
                domain=result["domain"],
                chunks=len(result["chunk_metas"]),
            )

        self._registry.save()
        log.info("init.complete", buckets=len(cluster_groups))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _generate_prose(self, data: dict) -> dict:
        """Call strategy.reason() with semaphore + jittered-exponential retry on 429."""
        digest = self._condenser.condense(data["cluster_chunks"], budget_tokens=_CONDENSER_BUDGET)
        prompt = (
            "Analyze this code and respond in EXACTLY this format:\n"
            "TITLE: <5-10 word aspect title>\n"
            "---\n"
            "<2-3 paragraph summary of what this code does, its main functions, and purpose>"
        )

        raw_response: str = ""
        for attempt in range(_MAX_RETRIES):
            async with self._sem:
                try:
                    raw_response = str(await self._strategy.reason(prompt, digest))
                    break
                except Exception as exc:
                    is_rate_limit = "429" in str(exc) or "rate_limit" in str(exc).lower()
                    if not is_rate_limit or attempt == _MAX_RETRIES - 1:
                        log.warning(
                            "init.prose_error",
                            bucket_id=data["bucket_id"],
                            attempt=attempt,
                            error=str(exc),
                        )
                        raw_response = ""
                        break
                    delay = _RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
                    log.warning(
                        "init.rate_limit_retry",
                        bucket_id=data["bucket_id"],
                        attempt=attempt,
                        delay=round(delay, 2),
                    )
                    await asyncio.sleep(delay)

        title, prose = _parse_title_and_prose(raw_response, data["fallback_label"])

        # Blend title embedding into centroid (α=0.2 title, 0.8 chunk mean).
        title_embed = self._embedder.embed(title).astype(np.float32)
        blended = (1 - _TITLE_BLEND_ALPHA) * data["centroid"] + _TITLE_BLEND_ALPHA * title_embed
        norm = np.linalg.norm(blended)
        if norm > 0:
            blended /= norm

        return {**data, "domain": title, "prose": prose, "centroid": blended}

    @staticmethod
    def _placeholder_prose(data: dict) -> dict:
        """Return the original static placeholder when no strategy is available."""
        domain = data["fallback_label"]
        chunk_metas = data["chunk_metas"]
        prose = f"# {domain}\n\nInitial seed — {len(chunk_metas)} chunks from: {domain}.\n"
        return {**data, "domain": domain, "prose": prose}
