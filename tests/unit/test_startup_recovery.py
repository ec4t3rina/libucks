"""Phase 6-B Testing Gate — test_startup_recovery.py

Tests the StartupRecovery logic that replays commits missed while the server
was offline. All git I/O is mocked at the module-level helper boundary so
the tests run without a real git repo.
"""
from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock, MagicMock, call, patch

import numpy as np
import pytest

from libucks.diff.diff_extractor import DiffExtractor
from libucks.librarian import Librarian
from libucks.models.chunk import ChunkMetadata
from libucks.models.events import DiffEvent, DiffHunk, UpdateEvent
from libucks.startup_recovery import StartupRecovery, _git_diff_name_only, _git_rev_parse_head
from libucks.storage.bucket_registry import BucketRegistry
from libucks.storage.bucket_store import BucketStore


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_OLD_SHA = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
_NEW_SHA = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
_BUCKET_ID = "deadbeef"

# A .py file tracked by the watcher extension set
_TRACKED_FILE = "libucks/auth.py"
# A .txt file NOT in the tracked extension set
_UNTRACKED_FILE = "README.txt"


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
    arr = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return base64.b64encode(arr.tobytes()).decode()


def _seed_bucket(store: BucketStore, bucket_id: str, source_file: str) -> None:
    """Create a minimal bucket file whose single chunk points at source_file."""
    chunk = ChunkMetadata(
        chunk_id="c001",
        source_file=source_file,
        start_line=1,
        end_line=10,
        git_sha=_OLD_SHA,
        token_count=50,
    )
    store.create(
        bucket_id=bucket_id,
        domain_label="test domain",
        centroid=_make_centroid_b64(),
        chunks=[chunk],
        prose="Some prose.",
    )


def _make_mock_librarian() -> AsyncMock:
    """Return an AsyncMock that mimics Librarian.handle()."""
    lib = AsyncMock(spec=Librarian)
    lib.bucket_id = _BUCKET_ID
    return lib


def _make_mock_extractor(events: List[DiffEvent]) -> MagicMock:
    """Return a MagicMock DiffExtractor whose extract_between() returns *events*."""
    ext = MagicMock(spec=DiffExtractor)
    ext.extract_between.return_value = events
    return ext


def _make_diff_event(filepath: str, added_lines: List[str] | None = None) -> DiffEvent:
    hunk = DiffHunk(
        file=filepath,
        old_start=1,
        old_end=5,
        new_start=1,
        new_end=6,
        added_lines=added_lines or ["def new_func(): pass"],
        removed_lines=[],
    )
    return DiffEvent(file=filepath, hunks=[hunk], is_rename=False)


def _make_recovery(
    tmp_path: Path,
    registry: BucketRegistry,
    store: BucketStore,
    librarians: dict,
    extractor: MagicMock,
) -> StartupRecovery:
    return StartupRecovery(
        repo_path=tmp_path,
        registry=registry,
        store=store,
        librarians=librarians,
        extractor=extractor,
    )


# ---------------------------------------------------------------------------
# Helper to run async tests without friction
# ---------------------------------------------------------------------------

_MOD = "libucks.startup_recovery"


# ---------------------------------------------------------------------------
# Tests: run() return value and baseline recording
# ---------------------------------------------------------------------------

class TestRunReturnValue:
    async def test_returns_current_head_when_up_to_date(
        self, tmp_path, tmp_registry, tmp_store
    ):
        tmp_registry._meta["last_indexed_head"] = _NEW_SHA
        recovery = _make_recovery(tmp_path, tmp_registry, tmp_store, {}, MagicMock())

        with patch(f"{_MOD}._git_rev_parse_head", return_value=_NEW_SHA), \
             patch(f"{_MOD}._git_diff_name_only", return_value=[]):
            result = await recovery.run()

        assert result == _NEW_SHA

    async def test_returns_none_when_git_unavailable(
        self, tmp_path, tmp_registry, tmp_store
    ):
        recovery = _make_recovery(tmp_path, tmp_registry, tmp_store, {}, MagicMock())

        with patch(f"{_MOD}._git_rev_parse_head", return_value=None):
            result = await recovery.run()

        assert result is None

    async def test_returns_current_head_when_no_baseline(
        self, tmp_path, tmp_registry, tmp_store
    ):
        # last_indexed_head is None (first boot after init)
        assert tmp_registry._meta.get("last_indexed_head") is None
        recovery = _make_recovery(tmp_path, tmp_registry, tmp_store, {}, MagicMock())

        with patch(f"{_MOD}._git_rev_parse_head", return_value=_NEW_SHA), \
             patch(f"{_MOD}._git_diff_name_only") as mock_diff:
            result = await recovery.run()

        assert result == _NEW_SHA
        # No diff call when there is no baseline to compare against
        mock_diff.assert_not_called()

    async def test_returns_current_head_after_successful_recovery(
        self, tmp_path, tmp_registry, tmp_store
    ):
        tmp_registry._meta["last_indexed_head"] = _OLD_SHA
        _seed_bucket(tmp_store, _BUCKET_ID, str(tmp_path / _TRACKED_FILE))
        lib = _make_mock_librarian()
        extractor = _make_mock_extractor([_make_diff_event(_TRACKED_FILE)])
        recovery = _make_recovery(tmp_path, tmp_registry, tmp_store, {_BUCKET_ID: lib}, extractor)

        with patch(f"{_MOD}._git_rev_parse_head", return_value=_NEW_SHA), \
             patch(f"{_MOD}._git_diff_name_only", return_value=[_TRACKED_FILE]):
            result = await recovery.run()

        assert result == _NEW_SHA


# ---------------------------------------------------------------------------
# Tests: recovery triggers librarian.handle()
# ---------------------------------------------------------------------------

class TestRecoveryUpdates:
    async def test_handle_called_for_changed_tracked_file(
        self, tmp_path, tmp_registry, tmp_store
    ):
        tmp_registry._meta["last_indexed_head"] = _OLD_SHA
        _seed_bucket(tmp_store, _BUCKET_ID, str(tmp_path / _TRACKED_FILE))
        lib = _make_mock_librarian()
        extractor = _make_mock_extractor([_make_diff_event(_TRACKED_FILE)])
        recovery = _make_recovery(tmp_path, tmp_registry, tmp_store, {_BUCKET_ID: lib}, extractor)

        with patch(f"{_MOD}._git_rev_parse_head", return_value=_NEW_SHA), \
             patch(f"{_MOD}._git_diff_name_only", return_value=[_TRACKED_FILE]):
            await recovery.run()

        lib.handle.assert_called_once()
        called_event = lib.handle.call_args[0][0]
        assert isinstance(called_event, UpdateEvent)
        assert called_event.bucket_id == _BUCKET_ID

    async def test_handle_called_once_per_hunk(
        self, tmp_path, tmp_registry, tmp_store
    ):
        """A DiffEvent with 3 hunks should produce 3 handle() calls."""
        tmp_registry._meta["last_indexed_head"] = _OLD_SHA
        _seed_bucket(tmp_store, _BUCKET_ID, str(tmp_path / _TRACKED_FILE))
        lib = _make_mock_librarian()

        hunk = DiffHunk(
            file=_TRACKED_FILE, old_start=1, old_end=2, new_start=1, new_end=3,
            added_lines=["x"], removed_lines=[],
        )
        multi_hunk_event = DiffEvent(
            file=_TRACKED_FILE, hunks=[hunk, hunk, hunk], is_rename=False
        )
        extractor = _make_mock_extractor([multi_hunk_event])
        recovery = _make_recovery(tmp_path, tmp_registry, tmp_store, {_BUCKET_ID: lib}, extractor)

        with patch(f"{_MOD}._git_rev_parse_head", return_value=_NEW_SHA), \
             patch(f"{_MOD}._git_diff_name_only", return_value=[_TRACKED_FILE]):
            await recovery.run()

        assert lib.handle.call_count == 3

    async def test_handle_not_called_when_head_unchanged(
        self, tmp_path, tmp_registry, tmp_store
    ):
        tmp_registry._meta["last_indexed_head"] = _NEW_SHA  # same as current
        lib = _make_mock_librarian()
        extractor = _make_mock_extractor([])
        recovery = _make_recovery(tmp_path, tmp_registry, tmp_store, {_BUCKET_ID: lib}, extractor)

        with patch(f"{_MOD}._git_rev_parse_head", return_value=_NEW_SHA), \
             patch(f"{_MOD}._git_diff_name_only", return_value=[]):
            await recovery.run()

        lib.handle.assert_not_called()

    async def test_handle_not_called_when_no_baseline(
        self, tmp_path, tmp_registry, tmp_store
    ):
        lib = _make_mock_librarian()
        extractor = _make_mock_extractor([])
        recovery = _make_recovery(tmp_path, tmp_registry, tmp_store, {_BUCKET_ID: lib}, extractor)

        with patch(f"{_MOD}._git_rev_parse_head", return_value=_NEW_SHA), \
             patch(f"{_MOD}._git_diff_name_only", return_value=[_TRACKED_FILE]):
            await recovery.run()

        lib.handle.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: file filtering
# ---------------------------------------------------------------------------

class TestFileFiltering:
    async def test_skips_file_with_untracked_extension(
        self, tmp_path, tmp_registry, tmp_store
    ):
        tmp_registry._meta["last_indexed_head"] = _OLD_SHA
        lib = _make_mock_librarian()
        extractor = _make_mock_extractor([_make_diff_event(_UNTRACKED_FILE)])
        recovery = _make_recovery(tmp_path, tmp_registry, tmp_store, {_BUCKET_ID: lib}, extractor)

        with patch(f"{_MOD}._git_rev_parse_head", return_value=_NEW_SHA), \
             patch(f"{_MOD}._git_diff_name_only", return_value=[_UNTRACKED_FILE]):
            await recovery.run()

        lib.handle.assert_not_called()
        extractor.extract_between.assert_not_called()

    async def test_skips_file_not_belonging_to_any_bucket(
        self, tmp_path, tmp_registry, tmp_store
    ):
        """A .py file that has no chunks in any bucket must be silently skipped."""
        tmp_registry._meta["last_indexed_head"] = _OLD_SHA
        # Note: we do NOT seed any bucket for _TRACKED_FILE here
        lib = _make_mock_librarian()
        extractor = _make_mock_extractor([_make_diff_event(_TRACKED_FILE)])
        recovery = _make_recovery(tmp_path, tmp_registry, tmp_store, {_BUCKET_ID: lib}, extractor)

        with patch(f"{_MOD}._git_rev_parse_head", return_value=_NEW_SHA), \
             patch(f"{_MOD}._git_diff_name_only", return_value=[_TRACKED_FILE]):
            await recovery.run()

        lib.handle.assert_not_called()

    async def test_handles_tracked_file_and_skips_untracked_in_same_diff(
        self, tmp_path, tmp_registry, tmp_store
    ):
        tmp_registry._meta["last_indexed_head"] = _OLD_SHA
        _seed_bucket(tmp_store, _BUCKET_ID, str(tmp_path / _TRACKED_FILE))
        lib = _make_mock_librarian()
        extractor = _make_mock_extractor([_make_diff_event(_TRACKED_FILE)])
        recovery = _make_recovery(tmp_path, tmp_registry, tmp_store, {_BUCKET_ID: lib}, extractor)

        changed = [_TRACKED_FILE, _UNTRACKED_FILE]
        with patch(f"{_MOD}._git_rev_parse_head", return_value=_NEW_SHA), \
             patch(f"{_MOD}._git_diff_name_only", return_value=changed):
            await recovery.run()

        # Tracked file: 1 handle call; untracked: skipped
        lib.handle.assert_called_once()

    async def test_skips_file_when_extractor_returns_empty_diff(
        self, tmp_path, tmp_registry, tmp_store
    ):
        """extract_between() returning [] (e.g. binary file) must not call handle."""
        tmp_registry._meta["last_indexed_head"] = _OLD_SHA
        _seed_bucket(tmp_store, _BUCKET_ID, str(tmp_path / _TRACKED_FILE))
        lib = _make_mock_librarian()
        extractor = _make_mock_extractor([])  # empty diff
        recovery = _make_recovery(tmp_path, tmp_registry, tmp_store, {_BUCKET_ID: lib}, extractor)

        with patch(f"{_MOD}._git_rev_parse_head", return_value=_NEW_SHA), \
             patch(f"{_MOD}._git_diff_name_only", return_value=[_TRACKED_FILE]):
            await recovery.run()

        lib.handle.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: extract_between receives correct arguments
# ---------------------------------------------------------------------------

class TestExtractorArgs:
    async def test_extract_between_called_with_correct_shas(
        self, tmp_path, tmp_registry, tmp_store
    ):
        tmp_registry._meta["last_indexed_head"] = _OLD_SHA
        _seed_bucket(tmp_store, _BUCKET_ID, str(tmp_path / _TRACKED_FILE))
        lib = _make_mock_librarian()
        extractor = _make_mock_extractor([_make_diff_event(_TRACKED_FILE)])
        recovery = _make_recovery(tmp_path, tmp_registry, tmp_store, {_BUCKET_ID: lib}, extractor)

        with patch(f"{_MOD}._git_rev_parse_head", return_value=_NEW_SHA), \
             patch(f"{_MOD}._git_diff_name_only", return_value=[_TRACKED_FILE]):
            await recovery.run()

        extractor.extract_between.assert_called_once()
        _, kwargs_or_args = extractor.extract_between.call_args[0], extractor.extract_between.call_args
        call_args = extractor.extract_between.call_args
        # Positional: (filepath, from_sha, to_sha)
        assert call_args[0][1] == _OLD_SHA   # from_sha
        assert call_args[0][2] == _NEW_SHA   # to_sha

    async def test_git_diff_name_only_called_with_correct_shas(
        self, tmp_path, tmp_registry, tmp_store
    ):
        tmp_registry._meta["last_indexed_head"] = _OLD_SHA
        recovery = _make_recovery(tmp_path, tmp_registry, tmp_store, {}, MagicMock())

        with patch(f"{_MOD}._git_rev_parse_head", return_value=_NEW_SHA), \
             patch(f"{_MOD}._git_diff_name_only", return_value=[]) as mock_diff:
            await recovery.run()

        mock_diff.assert_called_once_with(tmp_path, _OLD_SHA, _NEW_SHA)


# ---------------------------------------------------------------------------
# Tests: multiple files in the same diff
# ---------------------------------------------------------------------------

class TestMultipleFiles:
    async def test_two_tracked_files_in_different_buckets(
        self, tmp_path, tmp_registry, tmp_store
    ):
        tmp_registry._meta["last_indexed_head"] = _OLD_SHA

        file_a = "libucks/auth.py"
        file_b = "libucks/query.py"
        bucket_a = "aabbccdd"
        bucket_b = "eeff0011"

        _seed_bucket(tmp_store, bucket_a, str(tmp_path / file_a))
        _seed_bucket(tmp_store, bucket_b, str(tmp_path / file_b))

        lib_a = _make_mock_librarian()
        lib_b = _make_mock_librarian()

        # extractor returns appropriate events per file call
        extractor = MagicMock(spec=DiffExtractor)
        extractor.extract_between.side_effect = [
            [_make_diff_event(file_a)],
            [_make_diff_event(file_b)],
        ]

        recovery = _make_recovery(
            tmp_path, tmp_registry, tmp_store,
            {bucket_a: lib_a, bucket_b: lib_b},
            extractor,
        )

        with patch(f"{_MOD}._git_rev_parse_head", return_value=_NEW_SHA), \
             patch(f"{_MOD}._git_diff_name_only", return_value=[file_a, file_b]):
            await recovery.run()

        lib_a.handle.assert_called_once()
        lib_b.handle.assert_called_once()

    async def test_same_file_in_two_buckets_calls_both_librarians(
        self, tmp_path, tmp_registry, tmp_store
    ):
        """A file shared across two buckets should trigger updates in both."""
        tmp_registry._meta["last_indexed_head"] = _OLD_SHA

        bucket_a = "aabbccdd"
        bucket_b = "eeff0011"
        shared_file = _TRACKED_FILE

        _seed_bucket(tmp_store, bucket_a, str(tmp_path / shared_file))
        _seed_bucket(tmp_store, bucket_b, str(tmp_path / shared_file))

        lib_a = _make_mock_librarian()
        lib_b = _make_mock_librarian()
        extractor = _make_mock_extractor([_make_diff_event(shared_file)])

        recovery = _make_recovery(
            tmp_path, tmp_registry, tmp_store,
            {bucket_a: lib_a, bucket_b: lib_b},
            extractor,
        )

        with patch(f"{_MOD}._git_rev_parse_head", return_value=_NEW_SHA), \
             patch(f"{_MOD}._git_diff_name_only", return_value=[shared_file]):
            await recovery.run()

        lib_a.handle.assert_called_once()
        lib_b.handle.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: resilience
# ---------------------------------------------------------------------------

class TestResilience:
    async def test_extract_exception_skips_file_does_not_raise(
        self, tmp_path, tmp_registry, tmp_store
    ):
        tmp_registry._meta["last_indexed_head"] = _OLD_SHA
        _seed_bucket(tmp_store, _BUCKET_ID, str(tmp_path / _TRACKED_FILE))
        lib = _make_mock_librarian()

        extractor = MagicMock(spec=DiffExtractor)
        extractor.extract_between.side_effect = RuntimeError("git blew up")

        recovery = _make_recovery(tmp_path, tmp_registry, tmp_store, {_BUCKET_ID: lib}, extractor)

        with patch(f"{_MOD}._git_rev_parse_head", return_value=_NEW_SHA), \
             patch(f"{_MOD}._git_diff_name_only", return_value=[_TRACKED_FILE]):
            result = await recovery.run()  # must not raise

        assert result == _NEW_SHA
        lib.handle.assert_not_called()

    async def test_empty_changed_files_list_completes_cleanly(
        self, tmp_path, tmp_registry, tmp_store
    ):
        tmp_registry._meta["last_indexed_head"] = _OLD_SHA
        recovery = _make_recovery(tmp_path, tmp_registry, tmp_store, {}, MagicMock())

        with patch(f"{_MOD}._git_rev_parse_head", return_value=_NEW_SHA), \
             patch(f"{_MOD}._git_diff_name_only", return_value=[]):
            result = await recovery.run()

        assert result == _NEW_SHA
