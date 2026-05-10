"""Phase B.3 Integration Test — test_update_pipeline.py

Tests the full git commit → DiffExtractor → StartupRecovery → Librarian.handle()
chain using a real git repository and real DiffExtractor. No LLM calls are made —
Librarian is an AsyncMock.
"""
from __future__ import annotations

import base64
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest

from libucks.diff.diff_extractor import DiffExtractor
from libucks.librarian import Librarian
from libucks.models.chunk import ChunkMetadata
from libucks.models.events import UpdateEvent
from libucks.startup_recovery import StartupRecovery
from libucks.storage.bucket_registry import BucketRegistry
from libucks.storage.bucket_store import BucketStore


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git(cmd: list[str], cwd: Path) -> str:
    result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Minimal git repo with one Python file and an initial commit."""
    _git(["git", "init"], tmp_path)
    _git(["git", "config", "user.email", "test@libucks.test"], tmp_path)
    _git(["git", "config", "user.name", "libucks-test"], tmp_path)

    (tmp_path / "module.py").write_text("def hello():\n    return 'hello'\n")
    _git(["git", "add", "module.py"], tmp_path)
    _git(["git", "commit", "-m", "initial"], tmp_path)
    return tmp_path


@pytest.fixture
def libucks_dir(git_repo: Path) -> Path:
    d = git_repo / ".libucks"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Helpers: seed registry and store with one bucket owning module.py
# ---------------------------------------------------------------------------

_BUCKET_ID = "testbucket1"


def _seed(git_repo: Path, libucks_dir: Path) -> tuple[BucketRegistry, BucketStore, str]:
    """Seed registry (last_indexed_head = current HEAD) and bucket store."""
    head = _git(["git", "rev-parse", "HEAD"], git_repo)

    registry = BucketRegistry(libucks_dir / "registry.json")
    registry.load()
    registry._meta["last_indexed_head"] = head

    store = BucketStore(libucks_dir / "buckets")
    centroid = base64.b64encode(np.zeros(3, dtype=np.float32).tobytes()).decode()
    chunk = ChunkMetadata(
        chunk_id="c001",
        source_file=str(git_repo / "module.py"),
        start_line=1,
        end_line=2,
        git_sha=head,
        token_count=10,
    )
    store.create(
        bucket_id=_BUCKET_ID,
        domain_label="test domain",
        centroid=centroid,
        chunks=[chunk],
        prose="Original prose.",
    )
    return registry, store, head


def _make_recovery(
    git_repo: Path,
    registry: BucketRegistry,
    store: BucketStore,
    lib: AsyncMock,
) -> StartupRecovery:
    return StartupRecovery(
        repo_path=git_repo,
        registry=registry,
        store=store,
        librarians={_BUCKET_ID: lib},
        extractor=DiffExtractor(git_repo),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestUpdatePipeline:
    async def test_committed_change_triggers_update_event(
        self, git_repo: Path, libucks_dir: Path
    ):
        registry, store, _ = _seed(git_repo, libucks_dir)

        # Commit a change to module.py after recording the baseline
        (git_repo / "module.py").write_text("def hello():\n    return 'world'\n")
        _git(["git", "add", "module.py"], git_repo)
        _git(["git", "commit", "-m", "update hello"], git_repo)

        lib = AsyncMock(spec=Librarian)
        lib.bucket_id = _BUCKET_ID

        await _make_recovery(git_repo, registry, store, lib).run()

        lib.handle.assert_called()
        event = lib.handle.call_args[0][0]
        assert isinstance(event, UpdateEvent)
        assert event.bucket_id == _BUCKET_ID

    async def test_update_event_contains_added_line(
        self, git_repo: Path, libucks_dir: Path
    ):
        registry, store, _ = _seed(git_repo, libucks_dir)

        (git_repo / "module.py").write_text(
            "def hello():\n    return 'world'\n\ndef goodbye():\n    pass\n"
        )
        _git(["git", "add", "module.py"], git_repo)
        _git(["git", "commit", "-m", "add goodbye"], git_repo)

        lib = AsyncMock(spec=Librarian)
        lib.bucket_id = _BUCKET_ID

        await _make_recovery(git_repo, registry, store, lib).run()

        # At least one hunk must mention the new function
        all_added: list[str] = []
        for call in lib.handle.call_args_list:
            ev: UpdateEvent = call[0][0]
            all_added.extend(ev.hunk.added_lines)
        assert any("goodbye" in line for line in all_added)

    async def test_no_commits_since_baseline_skips_librarian(
        self, git_repo: Path, libucks_dir: Path
    ):
        registry, store, _ = _seed(git_repo, libucks_dir)
        # No new commits — HEAD matches last_indexed_head

        lib = AsyncMock(spec=Librarian)
        lib.bucket_id = _BUCKET_ID

        await _make_recovery(git_repo, registry, store, lib).run()

        lib.handle.assert_not_called()

    async def test_untracked_file_change_skips_librarian(
        self, git_repo: Path, libucks_dir: Path
    ):
        registry, store, _ = _seed(git_repo, libucks_dir)

        # Commit a change to a .txt file — not in _TRACKED_EXTENSIONS
        (git_repo / "README.txt").write_text("updated docs\n")
        _git(["git", "add", "README.txt"], git_repo)
        _git(["git", "commit", "-m", "update readme"], git_repo)

        lib = AsyncMock(spec=Librarian)
        lib.bucket_id = _BUCKET_ID

        await _make_recovery(git_repo, registry, store, lib).run()

        lib.handle.assert_not_called()

    async def test_returns_new_head_after_recovery(
        self, git_repo: Path, libucks_dir: Path
    ):
        registry, store, old_head = _seed(git_repo, libucks_dir)

        (git_repo / "module.py").write_text("def hello():\n    return 'changed'\n")
        _git(["git", "add", "module.py"], git_repo)
        _git(["git", "commit", "-m", "change"], git_repo)

        new_head = _git(["git", "rev-parse", "HEAD"], git_repo)

        lib = AsyncMock(spec=Librarian)
        lib.bucket_id = _BUCKET_ID

        result = await _make_recovery(git_repo, registry, store, lib).run()

        assert result == new_head
        assert result != old_head
