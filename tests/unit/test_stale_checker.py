"""Phase 6-C Testing Gate — test_stale_checker.py

Tests the four-level StaleChecker protocol and the QueryOrchestrator
integration (reindex_fn fired, eventual-consistency answer returned).

All OS and subprocess calls are patched at the module-level helper boundary
so the tests run without a real git repo or real files on disk.
"""
from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from libucks.models.chunk import ChunkMetadata
from libucks.stale_checker import StaleCheckResult, StaleChecker
from libucks.storage.bucket_registry import BucketRegistry
from libucks.storage.bucket_store import BucketStore

_MOD = "libucks.stale_checker"

_BUCKET_ID = "deadbeef"
_OTHER_ID = "cafebabe"
_OLD_SHA = "a" * 40
_NEW_SHA = "b" * 40
_SOURCE_FILE = "/repo/libucks/auth.py"
_INDEXED_AT = "2026-04-02T10:00:00+00:00"
from datetime import datetime as _dt
_INDEXED_TS: float = _dt.fromisoformat(_INDEXED_AT).timestamp()  # 1775124000.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_registry(tmp_path: Path) -> BucketRegistry:
    reg_path = tmp_path / ".libucks" / "registry.json"
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    return BucketRegistry(reg_path)


@pytest.fixture
def tmp_store(tmp_path: Path) -> BucketStore:
    return BucketStore(tmp_path / ".libucks" / "buckets")


def _make_centroid_b64() -> str:
    return base64.b64encode(np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes()).decode()


async def _seed_bucket(
    store: BucketStore,
    registry: BucketRegistry,
    bucket_id: str,
    source_file: str = _SOURCE_FILE,
    last_indexed_at: str | None = _INDEXED_AT,
    index_head_sha: str | None = _OLD_SHA,
) -> None:
    """Create a bucket with one chunk and populate the registry entry."""
    chunk = ChunkMetadata(
        chunk_id="c001",
        source_file=source_file,
        start_line=1,
        end_line=10,
        git_sha=index_head_sha or "init",
        token_count=50,
        indexed_at=last_indexed_at,
    )
    store.create(
        bucket_id=bucket_id,
        domain_label="test",
        centroid=_make_centroid_b64(),
        chunks=[chunk],
        prose="prose",
    )
    await registry.register(bucket_id, np.array([1.0, 0.0, 0.0], dtype=np.float32), 50)
    registry.update_index_timestamp(bucket_id, last_indexed_at or "", index_head_sha or "")


def _make_checker(
    registry: BucketRegistry,
    store: BucketStore,
    repo_path: Path,
) -> StaleChecker:
    return StaleChecker(registry=registry, store=store, repo_path=repo_path)


# ---------------------------------------------------------------------------
# Level 1 — watcher process alive?
# ---------------------------------------------------------------------------

class TestLevel1WatcherProcess:
    async def test_dead_watcher_pid_returns_stale_level_1(
        self, tmp_path, tmp_registry, tmp_store
    ):
        tmp_registry._meta["watcher_pid"] = 99999  # fake PID
        await _seed_bucket(tmp_store, tmp_registry, _BUCKET_ID)
        checker = _make_checker(tmp_registry, tmp_store, tmp_path)

        with patch(f"{_MOD}._process_is_alive", return_value=False), \
             patch(f"{_MOD}._get_current_head", return_value=_OLD_SHA):
            result = await checker.check([_BUCKET_ID])

        assert result.is_stale is True
        assert result.level == 1
        assert _BUCKET_ID in result.stale_bucket_ids

    async def test_alive_watcher_pid_does_not_trigger_level_1(
        self, tmp_path, tmp_registry, tmp_store
    ):
        tmp_registry._meta["watcher_pid"] = 12345
        tmp_registry._meta["last_indexed_head"] = _OLD_SHA
        await _seed_bucket(tmp_store, tmp_registry, _BUCKET_ID)
        checker = _make_checker(tmp_registry, tmp_store, tmp_path)

        with patch(f"{_MOD}._process_is_alive", return_value=True), \
             patch(f"{_MOD}._get_current_head", return_value=_OLD_SHA), \
             patch(f"{_MOD}._get_file_mtime", return_value=_INDEXED_TS - 1):
            result = await checker.check([_BUCKET_ID])

        assert result.is_stale is False
        assert result.level == 0

    async def test_no_watcher_pid_in_meta_skips_level_1(
        self, tmp_path, tmp_registry, tmp_store
    ):
        """If watcher_pid is None (never set), Level 1 is skipped entirely."""
        assert tmp_registry._meta.get("watcher_pid") is None
        tmp_registry._meta["last_indexed_head"] = _OLD_SHA
        await _seed_bucket(tmp_store, tmp_registry, _BUCKET_ID)
        checker = _make_checker(tmp_registry, tmp_store, tmp_path)

        with patch(f"{_MOD}._process_is_alive") as mock_kill, \
             patch(f"{_MOD}._get_current_head", return_value=_OLD_SHA), \
             patch(f"{_MOD}._get_file_mtime", return_value=_INDEXED_TS - 1):
            result = await checker.check([_BUCKET_ID])

        mock_kill.assert_not_called()
        assert result.is_stale is False

    async def test_level_1_flags_all_queried_buckets(
        self, tmp_path, tmp_registry, tmp_store
    ):
        tmp_registry._meta["watcher_pid"] = 99999
        await _seed_bucket(tmp_store, tmp_registry, _BUCKET_ID)
        await _seed_bucket(tmp_store, tmp_registry, _OTHER_ID, source_file="/repo/other.py")
        checker = _make_checker(tmp_registry, tmp_store, tmp_path)

        with patch(f"{_MOD}._process_is_alive", return_value=False):
            result = await checker.check([_BUCKET_ID, _OTHER_ID])

        assert result.is_stale is True
        assert set(result.stale_bucket_ids) == {_BUCKET_ID, _OTHER_ID}


# ---------------------------------------------------------------------------
# Level 2 — git HEAD drift
# ---------------------------------------------------------------------------

class TestLevel2HeadDrift:
    async def test_head_changed_returns_stale_level_2(
        self, tmp_path, tmp_registry, tmp_store
    ):
        tmp_registry._meta["last_indexed_head"] = _OLD_SHA
        await _seed_bucket(tmp_store, tmp_registry, _BUCKET_ID)
        checker = _make_checker(tmp_registry, tmp_store, tmp_path)

        with patch(f"{_MOD}._process_is_alive", return_value=True), \
             patch(f"{_MOD}._get_current_head", return_value=_NEW_SHA):
            result = await checker.check([_BUCKET_ID])

        assert result.is_stale is True
        assert result.level == 2
        assert _BUCKET_ID in result.stale_bucket_ids

    async def test_head_unchanged_does_not_trigger_level_2(
        self, tmp_path, tmp_registry, tmp_store
    ):
        tmp_registry._meta["last_indexed_head"] = _OLD_SHA
        await _seed_bucket(tmp_store, tmp_registry, _BUCKET_ID)
        checker = _make_checker(tmp_registry, tmp_store, tmp_path)

        with patch(f"{_MOD}._process_is_alive", return_value=True), \
             patch(f"{_MOD}._get_current_head", return_value=_OLD_SHA), \
             patch(f"{_MOD}._get_file_mtime", return_value=_INDEXED_TS - 1):
            result = await checker.check([_BUCKET_ID])

        assert result.level != 2

    async def test_level_2_flags_all_queried_buckets(
        self, tmp_path, tmp_registry, tmp_store
    ):
        tmp_registry._meta["last_indexed_head"] = _OLD_SHA
        await _seed_bucket(tmp_store, tmp_registry, _BUCKET_ID)
        await _seed_bucket(tmp_store, tmp_registry, _OTHER_ID, source_file="/repo/other.py")
        checker = _make_checker(tmp_registry, tmp_store, tmp_path)

        with patch(f"{_MOD}._process_is_alive", return_value=True), \
             patch(f"{_MOD}._get_current_head", return_value=_NEW_SHA):
            result = await checker.check([_BUCKET_ID, _OTHER_ID])

        assert result.is_stale is True
        assert set(result.stale_bucket_ids) == {_BUCKET_ID, _OTHER_ID}

    async def test_no_last_indexed_head_skips_level_2(
        self, tmp_path, tmp_registry, tmp_store
    ):
        """If last_indexed_head is None (never set), Level 2 cannot fire."""
        assert tmp_registry._meta.get("last_indexed_head") is None
        await _seed_bucket(tmp_store, tmp_registry, _BUCKET_ID)
        checker = _make_checker(tmp_registry, tmp_store, tmp_path)

        with patch(f"{_MOD}._process_is_alive", return_value=True), \
             patch(f"{_MOD}._get_current_head", return_value=_NEW_SHA), \
             patch(f"{_MOD}._get_file_mtime", return_value=_INDEXED_TS - 1):
            result = await checker.check([_BUCKET_ID])

        # Level 2 did not fire (no baseline to compare against)
        assert result.level != 2

    async def test_git_unavailable_skips_level_2(
        self, tmp_path, tmp_registry, tmp_store
    ):
        tmp_registry._meta["last_indexed_head"] = _OLD_SHA
        await _seed_bucket(tmp_store, tmp_registry, _BUCKET_ID)
        checker = _make_checker(tmp_registry, tmp_store, tmp_path)

        with patch(f"{_MOD}._process_is_alive", return_value=True), \
             patch(f"{_MOD}._get_current_head", return_value=None), \
             patch(f"{_MOD}._get_file_mtime", return_value=_INDEXED_TS - 1):
            result = await checker.check([_BUCKET_ID])

        # git unavailable → Level 2 skipped, still fresh
        assert result.level != 2


# ---------------------------------------------------------------------------
# Level 3 — file mtime vs last_indexed_at
# ---------------------------------------------------------------------------

class TestLevel3FileMtime:
    async def test_file_newer_than_index_triggers_level_3(
        self, tmp_path, tmp_registry, tmp_store
    ):
        tmp_registry._meta["last_indexed_head"] = _OLD_SHA
        await _seed_bucket(tmp_store, tmp_registry, _BUCKET_ID, last_indexed_at=_INDEXED_AT)
        checker = _make_checker(tmp_registry, tmp_store, tmp_path)

        with patch(f"{_MOD}._process_is_alive", return_value=True), \
             patch(f"{_MOD}._get_current_head", return_value=_OLD_SHA), \
             patch(f"{_MOD}._get_file_mtime", return_value=_INDEXED_TS + 60):  # 1 minute newer
            result = await checker.check([_BUCKET_ID])

        assert result.is_stale is True
        assert result.level == 3
        assert _BUCKET_ID in result.stale_bucket_ids

    async def test_file_older_than_index_is_fresh(
        self, tmp_path, tmp_registry, tmp_store
    ):
        tmp_registry._meta["last_indexed_head"] = _OLD_SHA
        await _seed_bucket(tmp_store, tmp_registry, _BUCKET_ID, last_indexed_at=_INDEXED_AT)
        checker = _make_checker(tmp_registry, tmp_store, tmp_path)

        with patch(f"{_MOD}._process_is_alive", return_value=True), \
             patch(f"{_MOD}._get_current_head", return_value=_OLD_SHA), \
             patch(f"{_MOD}._get_file_mtime", return_value=_INDEXED_TS - 60):  # older
            result = await checker.check([_BUCKET_ID])

        assert result.is_stale is False

    async def test_missing_last_indexed_at_skips_level_3(
        self, tmp_path, tmp_registry, tmp_store
    ):
        """Bucket with no last_indexed_at (pre-Phase-6-A) skips mtime check."""
        tmp_registry._meta["last_indexed_head"] = _OLD_SHA
        await _seed_bucket(
            tmp_store, tmp_registry, _BUCKET_ID,
            last_indexed_at=None, index_head_sha=_OLD_SHA,
        )
        checker = _make_checker(tmp_registry, tmp_store, tmp_path)

        with patch(f"{_MOD}._process_is_alive", return_value=True), \
             patch(f"{_MOD}._get_current_head", return_value=_OLD_SHA), \
             patch(f"{_MOD}._get_file_mtime") as mock_mtime:
            result = await checker.check([_BUCKET_ID])

        mock_mtime.assert_not_called()
        assert result.is_stale is False

    async def test_stat_failure_on_deleted_file_does_not_trigger_stale(
        self, tmp_path, tmp_registry, tmp_store
    ):
        """If a source file was deleted (stat returns None), skip that file."""
        tmp_registry._meta["last_indexed_head"] = _OLD_SHA
        await _seed_bucket(tmp_store, tmp_registry, _BUCKET_ID, last_indexed_at=_INDEXED_AT)
        checker = _make_checker(tmp_registry, tmp_store, tmp_path)

        with patch(f"{_MOD}._process_is_alive", return_value=True), \
             patch(f"{_MOD}._get_current_head", return_value=_OLD_SHA), \
             patch(f"{_MOD}._get_file_mtime", return_value=None):  # deleted
            result = await checker.check([_BUCKET_ID])

        assert result.is_stale is False

    async def test_level_3_deduplicates_source_files_across_chunks(
        self, tmp_path, tmp_registry, tmp_store
    ):
        """Two chunks from the same file should call _get_file_mtime only once."""
        from libucks.models.bucket import BucketFrontMatter

        chunk_a = ChunkMetadata(
            chunk_id="c001", source_file=_SOURCE_FILE,
            start_line=1, end_line=5, git_sha=_OLD_SHA, token_count=10,
            indexed_at=_INDEXED_AT,
        )
        chunk_b = ChunkMetadata(
            chunk_id="c002", source_file=_SOURCE_FILE,  # same file
            start_line=6, end_line=10, git_sha=_OLD_SHA, token_count=10,
            indexed_at=_INDEXED_AT,
        )
        tmp_store.create(
            bucket_id=_BUCKET_ID,
            domain_label="test",
            centroid=_make_centroid_b64(),
            chunks=[chunk_a, chunk_b],
            prose="prose",
        )
        await tmp_registry.register(_BUCKET_ID, np.array([1.0, 0.0, 0.0], dtype=np.float32), 20)
        tmp_registry.update_index_timestamp(_BUCKET_ID, _INDEXED_AT, _OLD_SHA)
        tmp_registry._meta["last_indexed_head"] = _OLD_SHA

        checker = _make_checker(tmp_registry, tmp_store, tmp_path)

        with patch(f"{_MOD}._process_is_alive", return_value=True), \
             patch(f"{_MOD}._get_current_head", return_value=_OLD_SHA), \
             patch(f"{_MOD}._get_file_mtime", return_value=_INDEXED_TS - 1) as mock_mtime:
            await checker.check([_BUCKET_ID])

        # Called once for the deduplicated source_file, not once per chunk
        mock_mtime.assert_called_once_with(_SOURCE_FILE)


# ---------------------------------------------------------------------------
# Level 4 — per-bucket index_head_sha drift
# ---------------------------------------------------------------------------

class TestLevel4BucketShaDrift:
    async def test_bucket_sha_behind_current_head_triggers_level_4(
        self, tmp_path, tmp_registry, tmp_store
    ):
        tmp_registry._meta["last_indexed_head"] = _NEW_SHA  # global HEAD is new
        await _seed_bucket(
            tmp_store, tmp_registry, _BUCKET_ID,
            last_indexed_at=_INDEXED_AT, index_head_sha=_OLD_SHA,  # bucket is behind
        )
        checker = _make_checker(tmp_registry, tmp_store, tmp_path)

        with patch(f"{_MOD}._process_is_alive", return_value=True), \
             patch(f"{_MOD}._get_current_head", return_value=_NEW_SHA), \
             patch(f"{_MOD}._get_file_mtime", return_value=_INDEXED_TS - 1):
            result = await checker.check([_BUCKET_ID])

        assert result.is_stale is True
        assert result.level == 4
        assert _BUCKET_ID in result.stale_bucket_ids

    async def test_bucket_sha_matches_current_head_is_fresh(
        self, tmp_path, tmp_registry, tmp_store
    ):
        tmp_registry._meta["last_indexed_head"] = _NEW_SHA
        await _seed_bucket(
            tmp_store, tmp_registry, _BUCKET_ID,
            last_indexed_at=_INDEXED_AT, index_head_sha=_NEW_SHA,
        )
        checker = _make_checker(tmp_registry, tmp_store, tmp_path)

        with patch(f"{_MOD}._process_is_alive", return_value=True), \
             patch(f"{_MOD}._get_current_head", return_value=_NEW_SHA), \
             patch(f"{_MOD}._get_file_mtime", return_value=_INDEXED_TS - 1):
            result = await checker.check([_BUCKET_ID])

        assert result.is_stale is False

    async def test_bucket_with_none_index_sha_skips_level_4(
        self, tmp_path, tmp_registry, tmp_store
    ):
        """Buckets without index_head_sha (pre-Phase-6-A) skip Level 4."""
        tmp_registry._meta["last_indexed_head"] = _NEW_SHA
        await _seed_bucket(
            tmp_store, tmp_registry, _BUCKET_ID,
            last_indexed_at=_INDEXED_AT, index_head_sha=None,
        )
        # Manually clear the index_head_sha on the entry (seed sets it to "")
        tmp_registry._buckets[_BUCKET_ID].index_head_sha = None
        checker = _make_checker(tmp_registry, tmp_store, tmp_path)

        with patch(f"{_MOD}._process_is_alive", return_value=True), \
             patch(f"{_MOD}._get_current_head", return_value=_NEW_SHA), \
             patch(f"{_MOD}._get_file_mtime", return_value=_INDEXED_TS - 1):
            result = await checker.check([_BUCKET_ID])

        assert result.is_stale is False

    async def test_level_4_only_flags_outdated_bucket_not_fresh_one(
        self, tmp_path, tmp_registry, tmp_store
    ):
        """Two buckets: one with current SHA, one with old SHA → only old one flagged."""
        tmp_registry._meta["last_indexed_head"] = _NEW_SHA

        await _seed_bucket(tmp_store, tmp_registry, _BUCKET_ID, index_head_sha=_OLD_SHA)
        await _seed_bucket(
            tmp_store, tmp_registry, _OTHER_ID,
            source_file="/repo/other.py", index_head_sha=_NEW_SHA,
        )
        checker = _make_checker(tmp_registry, tmp_store, tmp_path)

        with patch(f"{_MOD}._process_is_alive", return_value=True), \
             patch(f"{_MOD}._get_current_head", return_value=_NEW_SHA), \
             patch(f"{_MOD}._get_file_mtime", return_value=_INDEXED_TS - 1):
            result = await checker.check([_BUCKET_ID, _OTHER_ID])

        assert result.is_stale is True
        assert _BUCKET_ID in result.stale_bucket_ids
        assert _OTHER_ID not in result.stale_bucket_ids


# ---------------------------------------------------------------------------
# Level priority — higher levels don't fire when lower levels already returned
# ---------------------------------------------------------------------------

class TestLevelPriority:
    async def test_level_1_fires_before_level_2(
        self, tmp_path, tmp_registry, tmp_store
    ):
        tmp_registry._meta["watcher_pid"] = 99999
        tmp_registry._meta["last_indexed_head"] = _OLD_SHA
        await _seed_bucket(tmp_store, tmp_registry, _BUCKET_ID)
        checker = _make_checker(tmp_registry, tmp_store, tmp_path)

        with patch(f"{_MOD}._process_is_alive", return_value=False), \
             patch(f"{_MOD}._get_current_head", return_value=_NEW_SHA) as mock_head:
            result = await checker.check([_BUCKET_ID])

        assert result.level == 1
        # Level 2 subprocess call should NOT have been made since Level 1 returned early
        mock_head.assert_not_called()

    async def test_level_2_fires_before_level_3(
        self, tmp_path, tmp_registry, tmp_store
    ):
        tmp_registry._meta["last_indexed_head"] = _OLD_SHA
        await _seed_bucket(tmp_store, tmp_registry, _BUCKET_ID, last_indexed_at=_INDEXED_AT)
        checker = _make_checker(tmp_registry, tmp_store, tmp_path)

        with patch(f"{_MOD}._process_is_alive", return_value=True), \
             patch(f"{_MOD}._get_current_head", return_value=_NEW_SHA), \
             patch(f"{_MOD}._get_file_mtime") as mock_mtime:
            result = await checker.check([_BUCKET_ID])

        assert result.level == 2
        # Level 3 mtime check should NOT have been made since Level 2 returned early
        mock_mtime.assert_not_called()

    async def test_level_3_fires_before_level_4_for_same_bucket(
        self, tmp_path, tmp_registry, tmp_store
    ):
        """If Level 3 already flagged a bucket, Level 4 is not checked for that bucket."""
        tmp_registry._meta["last_indexed_head"] = _NEW_SHA
        await _seed_bucket(
            tmp_store, tmp_registry, _BUCKET_ID,
            last_indexed_at=_INDEXED_AT, index_head_sha=_OLD_SHA,
        )
        checker = _make_checker(tmp_registry, tmp_store, tmp_path)

        # File is newer (triggers L3) AND sha is old (would trigger L4)
        with patch(f"{_MOD}._process_is_alive", return_value=True), \
             patch(f"{_MOD}._get_current_head", return_value=_NEW_SHA), \
             patch(f"{_MOD}._get_file_mtime", return_value=_INDEXED_TS + 60):
            result = await checker.check([_BUCKET_ID])

        # Level 3 fires first
        assert result.level == 3


# ---------------------------------------------------------------------------
# Fresh result
# ---------------------------------------------------------------------------

class TestFreshResult:
    async def test_all_fresh_returns_is_stale_false(
        self, tmp_path, tmp_registry, tmp_store
    ):
        tmp_registry._meta["last_indexed_head"] = _OLD_SHA
        await _seed_bucket(
            tmp_store, tmp_registry, _BUCKET_ID,
            last_indexed_at=_INDEXED_AT, index_head_sha=_OLD_SHA,
        )
        checker = _make_checker(tmp_registry, tmp_store, tmp_path)

        with patch(f"{_MOD}._process_is_alive", return_value=True), \
             patch(f"{_MOD}._get_current_head", return_value=_OLD_SHA), \
             patch(f"{_MOD}._get_file_mtime", return_value=_INDEXED_TS - 1):
            result = await checker.check([_BUCKET_ID])

        assert result.is_stale is False
        assert result.level == 0
        assert result.stale_bucket_ids == []

    async def test_empty_bucket_ids_returns_fresh(
        self, tmp_path, tmp_registry, tmp_store
    ):
        tmp_registry._meta["last_indexed_head"] = _OLD_SHA
        checker = _make_checker(tmp_registry, tmp_store, tmp_path)

        with patch(f"{_MOD}._process_is_alive", return_value=True), \
             patch(f"{_MOD}._get_current_head", return_value=_OLD_SHA):
            result = await checker.check([])

        assert result.is_stale is False


# ---------------------------------------------------------------------------
# QueryOrchestrator integration
# ---------------------------------------------------------------------------

class TestQueryOrchestratorIntegration:
    """Verify the orchestrator fires reindex_fn on stale and still returns results."""

    def _make_orchestrator(self, stale_result: StaleCheckResult):
        from libucks.query_orchestrator import QueryOrchestrator
        from libucks.central_agent import CentralAgent

        # Minimal mocks — we just care about the stale-check branch
        mock_agent = MagicMock(spec=CentralAgent)
        mock_agent.route.return_value = [_BUCKET_ID]

        mock_librarian = AsyncMock()
        mock_librarian.handle.return_value = "some representation"

        mock_checker = AsyncMock()
        mock_checker.check.return_value = stale_result

        reindex_calls: list = []

        async def reindex_fn(bucket_ids):
            reindex_calls.append(bucket_ids)

        orch = QueryOrchestrator(
            central_agent=mock_agent,
            librarians={_BUCKET_ID: mock_librarian},
            embed_fn=lambda t: np.zeros(384, dtype=np.float32),
            top_k=1,
            stale_checker=mock_checker,
            reindex_fn=reindex_fn,
        )
        return orch, mock_checker, reindex_calls

    async def test_stale_result_fires_reindex_fn(self):
        stale = StaleCheckResult(
            is_stale=True,
            stale_bucket_ids=[_BUCKET_ID],
            level=3,
            reason="test stale",
        )
        orch, mock_checker, reindex_calls = self._make_orchestrator(stale)

        await orch.query("how does auth work?")

        # Give ensure_future time to schedule
        await asyncio.sleep(0)

        assert len(reindex_calls) == 1
        assert reindex_calls[0] == [_BUCKET_ID]

    async def test_stale_result_still_returns_answer(self):
        """Even when stale, the query must return results (eventual consistency)."""
        stale = StaleCheckResult(
            is_stale=True,
            stale_bucket_ids=[_BUCKET_ID],
            level=2,
            reason="HEAD drift",
        )
        orch, _, _ = self._make_orchestrator(stale)

        results = await orch.query("how does auth work?")

        assert results == ["some representation"]

    async def test_fresh_result_does_not_fire_reindex_fn(self):
        fresh = StaleCheckResult(
            is_stale=False,
            stale_bucket_ids=[],
            level=0,
            reason="fresh",
        )
        orch, mock_checker, reindex_calls = self._make_orchestrator(fresh)

        await orch.query("how does auth work?")
        await asyncio.sleep(0)

        assert reindex_calls == []

    async def test_no_stale_checker_query_still_works(self):
        """QueryOrchestrator without a stale_checker must work as before."""
        from libucks.query_orchestrator import QueryOrchestrator
        from libucks.central_agent import CentralAgent

        mock_agent = MagicMock(spec=CentralAgent)
        mock_agent.route.return_value = [_BUCKET_ID]
        mock_librarian = AsyncMock()
        mock_librarian.handle.return_value = "answer"

        orch = QueryOrchestrator(
            central_agent=mock_agent,
            librarians={_BUCKET_ID: mock_librarian},
            embed_fn=lambda t: np.zeros(384, dtype=np.float32),
            top_k=1,
            stale_checker=None,
            reindex_fn=None,
        )

        results = await orch.query("question?")
        assert results == ["answer"]
