"""Phase 1.5 Integration Test — test_novel_bucket_creation.py

Verifies that committing a file with novel + substantial content causes a new
bucket to be created (via StartupRecovery → create_bucket_queue → NovelBucketService),
while small or cosine-similar additions are absorbed into the nearest existing
bucket. Uses a real embedder (all-MiniLM-L6-v2) so novelty detection actually
exercises cosine distance; strategy + translator are mocked so we don't need
trained LoRA weights.
"""
from __future__ import annotations

import asyncio
import base64
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest

from libucks.central_agent import CentralAgent
from libucks.config import Config
from libucks.diff.diff_extractor import DiffExtractor
from libucks.embeddings.embedding_service import EmbeddingService
from libucks.librarian import Librarian
from libucks.models.chunk import ChunkMetadata
from libucks.models.events import CreateBucketEvent
from libucks.novel_bucket_service import NovelBucketService
from libucks.startup_recovery import StartupRecovery
from libucks.storage.bucket_registry import BucketRegistry
from libucks.storage.bucket_store import BucketStore


# ---------------------------------------------------------------------------
# Git helpers (lifted from test_update_pipeline.py to keep tests independent)
# ---------------------------------------------------------------------------

def _git(cmd: list[str], cwd: Path) -> str:
    result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Sample content — two semantically distinct domains
# ---------------------------------------------------------------------------

# Domain A: text-processing utilities. The initial bucket is seeded from this.
_DOMAIN_A_SEED = """\
def tokenize(text):
    \"\"\"Split text into words and punctuation tokens.\"\"\"
    import re
    return re.findall(r"\\w+|[^\\w\\s]", text)


def normalize(tokens):
    \"\"\"Lowercase and strip whitespace from each token.\"\"\"
    return [t.lower().strip() for t in tokens]


def word_frequencies(text):
    \"\"\"Return a dict mapping each token to its occurrence count.\"\"\"
    tokens = normalize(tokenize(text))
    counts = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    return counts
"""

# Domain B: SQL connection / transaction handling. Cosine-distant from domain A.
# Padded above the default min_bucket_seed_tokens=1500 (~6000 chars) to ensure
# substantial-novel content triggers bucket creation.
_DOMAIN_B_NOVEL = """\
import sqlite3
from contextlib import contextmanager
from typing import Iterator, Sequence, Any


class ConnectionPool:
    \"\"\"Lightweight connection pool for SQLite with WAL-mode by default.\"\"\"

    def __init__(self, database_path: str, max_connections: int = 5):
        self._path = database_path
        self._max = max_connections
        self._pool: list[sqlite3.Connection] = []
        self._in_use: set[sqlite3.Connection] = set()

    def _create_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def acquire(self) -> sqlite3.Connection:
        if self._pool:
            conn = self._pool.pop()
        elif len(self._in_use) < self._max:
            conn = self._create_connection()
        else:
            raise RuntimeError("connection pool exhausted")
        self._in_use.add(conn)
        return conn

    def release(self, conn: sqlite3.Connection) -> None:
        self._in_use.discard(conn)
        self._pool.append(conn)

    def close_all(self) -> None:
        for conn in list(self._pool) + list(self._in_use):
            try:
                conn.close()
            except Exception:
                pass
        self._pool.clear()
        self._in_use.clear()


@contextmanager
def transaction(pool: ConnectionPool) -> Iterator[sqlite3.Connection]:
    \"\"\"Acquire a connection, BEGIN a transaction, COMMIT or ROLLBACK on exit.\"\"\"
    conn = pool.acquire()
    try:
        conn.execute("BEGIN")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.release(conn)


def execute_many(
    pool: ConnectionPool,
    sql: str,
    rows: Sequence[Sequence[Any]],
) -> int:
    \"\"\"Run an INSERT or UPDATE for many parameter tuples inside one transaction.\"\"\"
    with transaction(pool) as conn:
        cursor = conn.executemany(sql, rows)
        return cursor.rowcount


def fetch_all(
    pool: ConnectionPool,
    sql: str,
    params: Sequence[Any] = (),
) -> list[sqlite3.Row]:
    \"\"\"Run a SELECT and return all rows as sqlite3.Row objects.\"\"\"
    conn = pool.acquire()
    try:
        cursor = conn.execute(sql, params)
        return cursor.fetchall()
    finally:
        pool.release(conn)
"""

# Domain A small addition: more text-processing helpers. Similar to the seed,
# so cosine novelty should NOT fire. Also kept short so even if it were novel,
# the size guard would absorb it.
_DOMAIN_A_SMALL = """\
def is_alpha(token):
    return token.isalpha()
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def embedder() -> EmbeddingService:
    return EmbeddingService.get_instance("all-MiniLM-L6-v2")


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Minimal git repo seeded with one file in domain A and one initial commit."""
    _git(["git", "init"], tmp_path)
    _git(["git", "config", "user.email", "test@libucks.test"], tmp_path)
    _git(["git", "config", "user.name", "libucks-test"], tmp_path)

    (tmp_path / "text_utils.py").write_text(_DOMAIN_A_SEED)
    _git(["git", "add", "text_utils.py"], tmp_path)
    _git(["git", "commit", "-m", "initial: text utilities"], tmp_path)
    return tmp_path


@pytest.fixture
def libucks_dir(git_repo: Path) -> Path:
    d = git_repo / ".libucks"
    d.mkdir()
    return d


async def _seed_initial_bucket(
    git_repo: Path,
    libucks_dir: Path,
    embedder: EmbeddingService,
) -> tuple[BucketRegistry, BucketStore, str, str]:
    """Seed one real bucket from text_utils.py so routing has somewhere to land."""
    head = _git(["git", "rev-parse", "HEAD"], git_repo)

    registry = BucketRegistry(libucks_dir / "registry.json")
    registry.load()
    registry._meta["last_indexed_head"] = head

    store = BucketStore(libucks_dir / "buckets")

    centroid_vec = embedder.embed(_DOMAIN_A_SEED).astype(np.float32)
    centroid_vec /= np.linalg.norm(centroid_vec) or 1.0
    centroid_b64 = base64.b64encode(centroid_vec.tobytes()).decode()

    bucket_id = "textutil1"
    chunk = ChunkMetadata(
        chunk_id="c001",
        source_file=str(git_repo / "text_utils.py"),
        start_line=1,
        end_line=len(_DOMAIN_A_SEED.splitlines()),
        git_sha=head,
        token_count=200,
    )
    store.create(
        bucket_id=bucket_id,
        domain_label="text utilities",
        centroid=centroid_b64,
        chunks=[chunk],
        prose="Initial prose for text utilities.",
    )
    await registry.register(bucket_id, centroid_vec, 200)
    return registry, store, head, bucket_id


def _make_central_agent(
    registry: BucketRegistry,
    embedder: EmbeddingService,
) -> CentralAgent:
    return CentralAgent(registry, Config(), embed_fn=embedder.embed)


# Lower threshold for tests so we can exercise the size guard with manageable
# content (~2.4 KB → ~600 tokens). Production default remains 1500 per config.
_TEST_MIN_SEED_TOKENS = 200


def _make_recovery(
    git_repo: Path,
    registry: BucketRegistry,
    store: BucketStore,
    librarians: dict,
    agent: CentralAgent,
    embedder: EmbeddingService,
    min_bucket_seed_tokens: int = _TEST_MIN_SEED_TOKENS,
) -> StartupRecovery:
    return StartupRecovery(
        repo_path=git_repo,
        registry=registry,
        store=store,
        librarians=librarians,
        extractor=DiffExtractor(git_repo),
        central_agent=agent,
        embedder=embedder,
        min_bucket_seed_tokens=min_bucket_seed_tokens,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNovelBucketProducer:
    """StartupRecovery enqueues CreateBucketEvent for substantial novel files
    and routes everything else to the nearest existing bucket."""

    async def test_substantial_novel_file_enqueues_create_event(
        self,
        git_repo: Path,
        libucks_dir: Path,
        embedder: EmbeddingService,
    ):
        registry, store, _, bucket_id = await _seed_initial_bucket(git_repo, libucks_dir, embedder)
        lib = AsyncMock(spec=Librarian)
        lib.bucket_id = bucket_id
        agent = _make_central_agent(registry, embedder)
        agent.register_librarian(bucket_id, lib)

        (git_repo / "db_pool.py").write_text(_DOMAIN_B_NOVEL)
        _git(["git", "add", "db_pool.py"], git_repo)
        _git(["git", "commit", "-m", "add db pool"], git_repo)

        await _make_recovery(git_repo, registry, store, {bucket_id: lib}, agent, embedder).run()

        assert agent.create_bucket_queue.qsize() == 1
        event = agent.create_bucket_queue.get_nowait()
        assert isinstance(event, CreateBucketEvent)
        assert event.source_file == "db_pool.py"
        assert "ConnectionPool" in event.seed_content
        # Producer should NOT have also routed the file to the existing bucket.
        lib.handle.assert_not_called()

    async def test_small_novel_file_routes_to_existing_bucket(
        self,
        git_repo: Path,
        libucks_dir: Path,
        embedder: EmbeddingService,
    ):
        registry, store, _, bucket_id = await _seed_initial_bucket(git_repo, libucks_dir, embedder)
        lib = AsyncMock(spec=Librarian)
        lib.bucket_id = bucket_id
        agent = _make_central_agent(registry, embedder)
        agent.register_librarian(bucket_id, lib)

        # Cosine-distant but tiny: even though it'd be "novel", the size guard
        # should absorb it into the nearest existing bucket.
        small_novel = "import sqlite3\n\nconn = sqlite3.connect(':memory:')\n"
        (git_repo / "tiny_db.py").write_text(small_novel)
        _git(["git", "add", "tiny_db.py"], git_repo)
        _git(["git", "commit", "-m", "add tiny db"], git_repo)

        await _make_recovery(git_repo, registry, store, {bucket_id: lib}, agent, embedder).run()

        assert agent.create_bucket_queue.qsize() == 0
        lib.handle.assert_called()  # routed to the only existing bucket

    async def test_similar_addition_routes_to_existing_bucket(
        self,
        git_repo: Path,
        libucks_dir: Path,
        embedder: EmbeddingService,
    ):
        registry, store, _, bucket_id = await _seed_initial_bucket(git_repo, libucks_dir, embedder)
        lib = AsyncMock(spec=Librarian)
        lib.bucket_id = bucket_id
        agent = _make_central_agent(registry, embedder)
        agent.register_librarian(bucket_id, lib)

        # Same domain as the seed — should NOT be considered novel even if large.
        (git_repo / "more_text_utils.py").write_text(_DOMAIN_A_SEED + _DOMAIN_A_SEED)
        _git(["git", "add", "more_text_utils.py"], git_repo)
        _git(["git", "commit", "-m", "extend text utils"], git_repo)

        await _make_recovery(git_repo, registry, store, {bucket_id: lib}, agent, embedder).run()

        assert agent.create_bucket_queue.qsize() == 0
        lib.handle.assert_called()


class TestNovelBucketConsumer:
    """NovelBucketService drains create_bucket_queue and creates a real bucket
    that subsequent commits to the same file can find."""

    async def test_consumer_creates_bucket_and_makes_it_findable(
        self,
        git_repo: Path,
        libucks_dir: Path,
        embedder: EmbeddingService,
    ):
        registry, store, _, seed_bucket = await _seed_initial_bucket(git_repo, libucks_dir, embedder)
        lib = AsyncMock(spec=Librarian)
        lib.bucket_id = seed_bucket
        agent = _make_central_agent(registry, embedder)
        agent.register_librarian(seed_bucket, lib)

        (git_repo / "db_pool.py").write_text(_DOMAIN_B_NOVEL)
        _git(["git", "add", "db_pool.py"], git_repo)
        _git(["git", "commit", "-m", "add db pool"], git_repo)

        recovery = _make_recovery(git_repo, registry, store, {seed_bucket: lib}, agent, embedder)
        await recovery.run()
        assert agent.create_bucket_queue.qsize() == 1

        # Run the consumer. strategy + translator are None so prose generation
        # falls back to a placeholder — we don't need trained weights for this test.
        consumer = NovelBucketService(
            store=store,
            registry=registry,
            embedder=embedder,
            agent=agent,
            strategy=None,
            translator=None,
            repo_path=git_repo,
        )
        await consumer.drain_pending()

        # Registry now has 2 buckets: the original + the newly-created one.
        all_bucket_ids = list(registry.get_all_centroids().keys())
        assert len(all_bucket_ids) == 2
        new_bucket_id = [b for b in all_bucket_ids if b != seed_bucket][0]

        # New bucket is registered with a Librarian on the agent.
        assert new_bucket_id in agent._librarians

        # The newly-created bucket's source_file matches the new file —
        # so a subsequent edit will be picked up by _find_buckets_for_file.
        front_matter, _ = store.read(new_bucket_id)
        assert any(c.source_file == "db_pool.py" for c in front_matter.chunks)

        # A follow-up commit to the same file routes back to the new bucket
        # (not to the original text-utilities bucket) — proves the
        # "self-evolving on second commit" claim.
        new_lib = AsyncMock(spec=Librarian)
        new_lib.bucket_id = new_bucket_id
        agent.register_librarian(new_bucket_id, new_lib)

        (git_repo / "db_pool.py").write_text(_DOMAIN_B_NOVEL + "\n# update\n")
        _git(["git", "add", "db_pool.py"], git_repo)
        _git(["git", "commit", "-m", "update db pool"], git_repo)

        # Re-baseline the registry head to current and re-run recovery so the
        # second commit gets replayed.
        registry._meta["last_indexed_head"] = _git(
            ["git", "rev-parse", "HEAD~1"], git_repo
        )
        recovery2 = _make_recovery(
            git_repo, registry, store,
            {seed_bucket: lib, new_bucket_id: new_lib},
            agent, embedder,
        )
        await recovery2.run()

        # The follow-up commit went to the new bucket, not the original.
        assert new_lib.handle.called
