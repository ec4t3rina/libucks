"""Phase 2 Testing Gate — test_central_agent_events.py

Tests the async CentralAgent event loop:
  - DiffEvent with added lines  → UpdateEvent (or CreateBucketEvent if novel)
  - DiffEvent with removed lines → TombstoneEvent
  - DiffEvent with is_rename=True → PathUpdateEvent broadcast to all librarians
  - Mitosis guard: event buffered when is_splitting=True, drained when cleared

No embedding model is loaded. A deterministic embed_fn is injected so routing
is fully controlled by unit-vector centroids.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from libucks.central_agent import CentralAgent
from libucks.config import Config
from libucks.librarian import Librarian
from libucks.models.events import (
    CreateBucketEvent,
    DiffEvent,
    DiffHunk,
    PathUpdateEvent,
    TombstoneEvent,
    UpdateEvent,
)
from libucks.storage.bucket_registry import BucketRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vec(n: int, dim: int = 384) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[n] = 1.0
    return v


def _embed_at(n: int):
    """Returns an embed_fn that always produces _unit_vec(n)."""
    return lambda _text: _unit_vec(n)


def _add_hunk(file: str, added: list[str]) -> DiffHunk:
    return DiffHunk(
        file=file,
        old_start=0, old_end=0,
        new_start=1, new_end=len(added),
        added_lines=added,
        removed_lines=[],
    )


def _del_hunk(file: str, removed: list[str]) -> DiffHunk:
    return DiffHunk(
        file=file,
        old_start=1, old_end=len(removed),
        new_start=0, new_end=0,
        added_lines=[],
        removed_lines=removed,
    )


def _add_event(file: str, added: list[str]) -> DiffEvent:
    return DiffEvent(file=file, hunks=[_add_hunk(file, added)], is_rename=False)


def _del_event(file: str, removed: list[str]) -> DiffEvent:
    return DiffEvent(file=file, hunks=[_del_hunk(file, removed)], is_rename=False)


def _rename_event(old: str, new: str) -> DiffEvent:
    return DiffEvent(file=new, hunks=[], is_rename=True, old_path=old, new_path=new)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def registry(tmp_path):
    return BucketRegistry(tmp_path / "registry.json")


@pytest.fixture
def config():
    return Config()


# ---------------------------------------------------------------------------
# UPDATE path — added lines
# ---------------------------------------------------------------------------

class TestAddedLines:
    async def test_added_lines_dispatches_update_event(self, registry, config):
        """Added hunk routes to the nearest bucket → UpdateEvent delivered."""
        await registry.register("bucket_a", _unit_vec(0), 100)
        lib_a = Librarian("bucket_a")

        agent = CentralAgent(registry=registry, config=config, embed_fn=_embed_at(0))
        agent.register_librarian("bucket_a", lib_a)

        await agent.post(_add_event("src/auth.py", ["def login(): pass"]))
        await agent.run_once()

        assert lib_a.queue.qsize() == 1
        event = lib_a.queue.get_nowait()
        assert isinstance(event, UpdateEvent)
        assert event.bucket_id == "bucket_a"

    async def test_update_event_carries_correct_hunk(self, registry, config):
        """The UpdateEvent wraps the original DiffHunk."""
        await registry.register("bucket_a", _unit_vec(0), 100)
        lib_a = Librarian("bucket_a")

        agent = CentralAgent(registry=registry, config=config, embed_fn=_embed_at(0))
        agent.register_librarian("bucket_a", lib_a)

        added = ["def login(): pass", "    return True"]
        await agent.post(_add_event("src/auth.py", added))
        await agent.run_once()

        event: UpdateEvent = lib_a.queue.get_nowait()
        assert event.hunk.added_lines == added

    async def test_routes_to_closest_bucket(self, registry, config):
        """With two buckets, the event goes to the one with the closer centroid."""
        await registry.register("bucket_a", _unit_vec(0), 100)
        await registry.register("bucket_b", _unit_vec(1), 100)
        lib_a = Librarian("bucket_a")
        lib_b = Librarian("bucket_b")

        # embed_fn returns _unit_vec(1) → closer to bucket_b
        agent = CentralAgent(registry=registry, config=config, embed_fn=_embed_at(1))
        agent.register_librarian("bucket_a", lib_a)
        agent.register_librarian("bucket_b", lib_b)

        await agent.post(_add_event("src/db.py", ["class User(Base): pass"]))
        await agent.run_once()

        assert lib_b.queue.qsize() == 1
        assert lib_a.queue.qsize() == 0


# ---------------------------------------------------------------------------
# NOVEL path — no close centroid
# ---------------------------------------------------------------------------

class TestNovelDiff:
    async def test_novel_diff_emits_create_bucket_event(self, registry, config):
        """Empty registry → every diff is novel → CreateBucketEvent."""
        agent = CentralAgent(registry=registry, config=config, embed_fn=_embed_at(0))

        await agent.post(_add_event("src/new.py", ["def brand_new(): pass"]))
        await agent.run_once()

        assert agent.create_bucket_queue.qsize() == 1
        event = agent.create_bucket_queue.get_nowait()
        assert isinstance(event, CreateBucketEvent)
        assert "brand_new" in event.seed_content

    async def test_novel_diff_not_delivered_to_any_librarian(self, registry, config):
        """Novel diff → CreateBucketEvent only, no librarian receives anything."""
        lib = Librarian("bucket_a")
        agent = CentralAgent(registry=registry, config=config, embed_fn=_embed_at(0))
        agent.register_librarian("bucket_a", lib)  # registered but empty registry

        await agent.post(_add_event("src/new.py", ["def brand_new(): pass"]))
        await agent.run_once()

        assert lib.queue.qsize() == 0


# ---------------------------------------------------------------------------
# TOMBSTONE path — deleted lines
# ---------------------------------------------------------------------------

class TestDeletedLines:
    async def test_deleted_lines_dispatches_tombstone_event(self, registry, config):
        """A deletion hunk → TombstoneEvent to the routed bucket."""
        await registry.register("bucket_a", _unit_vec(0), 100)
        lib_a = Librarian("bucket_a")

        agent = CentralAgent(registry=registry, config=config, embed_fn=_embed_at(0))
        agent.register_librarian("bucket_a", lib_a)

        await agent.post(_del_event("src/auth.py", ["def old_func(): pass"]))
        await agent.run_once()

        assert lib_a.queue.qsize() == 1
        event = lib_a.queue.get_nowait()
        assert isinstance(event, TombstoneEvent)
        assert "bucket_a" in event.bucket_ids

    async def test_deleted_lines_do_not_produce_update_event(self, registry, config):
        """Pure deletion must not generate an UpdateEvent."""
        await registry.register("bucket_a", _unit_vec(0), 100)
        lib_a = Librarian("bucket_a")

        agent = CentralAgent(registry=registry, config=config, embed_fn=_embed_at(0))
        agent.register_librarian("bucket_a", lib_a)

        await agent.post(_del_event("src/auth.py", ["def old_func(): pass"]))
        await agent.run_once()

        event = lib_a.queue.get_nowait()
        assert not isinstance(event, UpdateEvent)


# ---------------------------------------------------------------------------
# RENAME path
# ---------------------------------------------------------------------------

class TestRenameEvent:
    async def test_rename_dispatches_path_update_event(self, registry, config):
        """Rename → PathUpdateEvent broadcast to every registered librarian."""
        await registry.register("bucket_a", _unit_vec(0), 100)
        await registry.register("bucket_b", _unit_vec(1), 100)
        lib_a = Librarian("bucket_a")
        lib_b = Librarian("bucket_b")

        agent = CentralAgent(registry=registry, config=config, embed_fn=_embed_at(0))
        agent.register_librarian("bucket_a", lib_a)
        agent.register_librarian("bucket_b", lib_b)

        await agent.post(_rename_event("src/auth.py", "src/authentication.py"))
        await agent.run_once()

        assert lib_a.queue.qsize() == 1
        assert lib_b.queue.qsize() == 1

    async def test_rename_paths_are_correct(self, registry, config):
        """PathUpdateEvent carries the exact old and new paths."""
        await registry.register("bucket_a", _unit_vec(0), 100)
        lib_a = Librarian("bucket_a")

        agent = CentralAgent(registry=registry, config=config, embed_fn=_embed_at(0))
        agent.register_librarian("bucket_a", lib_a)

        await agent.post(_rename_event("src/auth.py", "src/authentication.py"))
        await agent.run_once()

        event: PathUpdateEvent = lib_a.queue.get_nowait()
        assert isinstance(event, PathUpdateEvent)
        assert event.old_path == "src/auth.py"
        assert event.new_path == "src/authentication.py"

    async def test_rename_does_not_produce_update_or_tombstone(self, registry, config):
        """Rename must not generate UpdateEvent or TombstoneEvent."""
        await registry.register("bucket_a", _unit_vec(0), 100)
        lib_a = Librarian("bucket_a")

        agent = CentralAgent(registry=registry, config=config, embed_fn=_embed_at(0))
        agent.register_librarian("bucket_a", lib_a)

        await agent.post(_rename_event("old.py", "new.py"))
        await agent.run_once()

        event = lib_a.queue.get_nowait()
        assert not isinstance(event, (UpdateEvent, TombstoneEvent))


# ---------------------------------------------------------------------------
# Mitosis guard
# ---------------------------------------------------------------------------

class TestMitosisGuard:
    async def test_event_buffered_when_splitting(self, registry, config):
        """While is_splitting=True, the UpdateEvent goes to the retry buffer."""
        await registry.register("bucket_a", _unit_vec(0), 100)
        await registry.set_splitting("bucket_a", True)
        lib_a = Librarian("bucket_a")

        agent = CentralAgent(registry=registry, config=config, embed_fn=_embed_at(0))
        agent.register_librarian("bucket_a", lib_a)

        await agent.post(_add_event("src/auth.py", ["def login(): pass"]))
        await agent.run_once()

        assert lib_a.queue.qsize() == 0
        assert len(agent.retry_buffer) == 1

    async def test_buffered_event_not_lost(self, registry, config):
        """The retry buffer entry refers to the correct bucket and event type."""
        await registry.register("bucket_a", _unit_vec(0), 100)
        await registry.set_splitting("bucket_a", True)
        lib_a = Librarian("bucket_a")

        agent = CentralAgent(registry=registry, config=config, embed_fn=_embed_at(0))
        agent.register_librarian("bucket_a", lib_a)

        await agent.post(_add_event("src/auth.py", ["def login(): pass"]))
        await agent.run_once()

        bucket_id, buffered_event = agent.retry_buffer[0]
        assert bucket_id == "bucket_a"
        assert isinstance(buffered_event, UpdateEvent)

    async def test_event_delivered_after_splitting_clears(self, registry, config):
        """After clear_splitting(), buffered events are drained to the librarian."""
        await registry.register("bucket_a", _unit_vec(0), 100)
        await registry.set_splitting("bucket_a", True)
        lib_a = Librarian("bucket_a")

        agent = CentralAgent(registry=registry, config=config, embed_fn=_embed_at(0))
        agent.register_librarian("bucket_a", lib_a)

        await agent.post(_add_event("src/auth.py", ["def login(): pass"]))
        await agent.run_once()
        assert lib_a.queue.qsize() == 0  # not yet delivered

        await agent.clear_splitting("bucket_a")

        assert lib_a.queue.qsize() == 1
        event = lib_a.queue.get_nowait()
        assert isinstance(event, UpdateEvent)
        assert event.bucket_id == "bucket_a"

    async def test_retry_buffer_empty_after_drain(self, registry, config):
        """Retry buffer is fully drained once clear_splitting() is called."""
        await registry.register("bucket_a", _unit_vec(0), 100)
        await registry.set_splitting("bucket_a", True)
        lib_a = Librarian("bucket_a")

        agent = CentralAgent(registry=registry, config=config, embed_fn=_embed_at(0))
        agent.register_librarian("bucket_a", lib_a)

        # Post two events while splitting
        await agent.post(_add_event("src/auth.py", ["def login(): pass"]))
        await agent.post(_add_event("src/auth.py", ["def logout(): pass"]))
        await agent.run_once()
        await agent.run_once()
        assert len(agent.retry_buffer) == 2

        await agent.clear_splitting("bucket_a")

        assert len(agent.retry_buffer) == 0
        assert lib_a.queue.qsize() == 2
