"""Phase D: Librarian — translator injection for prose generation.

Verifies that:
  - _handle_update passes the tensor from strategy.reason() through
    translator.synthesize() and writes the decoded string as prose.
  - _handle_tombstone does the same when surviving chunks remain.
  - When translator is None, prose is NOT regenerated (no str(tensor) written).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from unittest.mock import AsyncMock, MagicMock, patch

from libucks.librarian import Librarian
from libucks.models.bucket import BucketFrontMatter
from libucks.models.chunk import ChunkMetadata
from libucks.models.events import DiffHunk, TombstoneEvent, UpdateEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_front_matter(bucket_id: str = "b1") -> BucketFrontMatter:
    chunk = ChunkMetadata(
        chunk_id="c1",
        source_file="/fake/auth.py",
        start_line=1,
        end_line=5,
        git_sha="deadbeef",
        token_count=100,
    )
    return BucketFrontMatter(
        bucket_id=bucket_id,
        domain_label="auth",
        centroid_embedding="AAAA",
        token_count=100,
        chunks=[chunk],
    )


def _make_lock():
    lock = MagicMock()
    lock.__aenter__ = AsyncMock(return_value=None)
    lock.__aexit__ = AsyncMock(return_value=False)
    return lock


def _make_update_event(bucket_id: str = "b1") -> UpdateEvent:
    hunk = DiffHunk(
        file="/fake/auth.py",
        old_start=1, old_end=2,
        new_start=1, new_end=3,
        added_lines=["def login(): pass"],
        removed_lines=[],
    )
    return UpdateEvent(bucket_id=bucket_id, hunk=hunk)


def _make_tombstone_event(chunk_ids: list[str]) -> TombstoneEvent:
    # Leave one chunk surviving so prose regeneration is triggered.
    return TombstoneEvent(chunk_ids=chunk_ids, bucket_ids=["b1"])


def _make_mock_registry() -> MagicMock:
    registry = MagicMock()
    registry.get_lock.return_value = _make_lock()
    registry.register = AsyncMock()
    registry.update_index_timestamp = MagicMock()
    registry.get_all_centroids.return_value = {}
    registry.save = MagicMock()
    return registry


def _make_mock_embedder(dim: int = 8) -> MagicMock:
    embedder = MagicMock()
    embedder.embed_batch.return_value = np.random.rand(1, dim).astype(np.float32)
    embedder.embed.return_value = np.random.rand(dim).astype(np.float32)
    return embedder


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestLibrarianTranslatorConstruction:
    def test_translator_defaults_to_none(self):
        lib = Librarian("b1")
        assert lib._translator is None

    def test_translator_stored_when_injected(self):
        translator = MagicMock()
        lib = Librarian("b1", translator=translator)
        assert lib._translator is translator


# ---------------------------------------------------------------------------
# _handle_update
# ---------------------------------------------------------------------------

class TestLibrarianUpdateProse:
    async def test_update_calls_translator_synthesize_with_tensor(self):
        sentinel = torch.ones(4, 8)
        strategy = MagicMock()
        strategy.reason = AsyncMock(return_value=sentinel)

        translator = MagicMock()
        translator.synthesize = AsyncMock(return_value="decoded prose")

        store = MagicMock()
        store.read.return_value = (_make_front_matter(), "old prose")
        store.write_prose = MagicMock()
        store.write_front_matter = MagicMock()

        lib = Librarian(
            "b1",
            store=store,
            registry=_make_mock_registry(),
            strategy=strategy,
            embedder=_make_mock_embedder(),
            translator=translator,
        )

        with patch("libucks.librarian._read_chunk_content", return_value="def foo(): pass"):
            await lib._handle_update(_make_update_event())

        call_args = translator.synthesize.call_args
        assert call_args[0][0] == ""           # first arg: query string
        reps = call_args[0][1]
        assert len(reps) == 1
        assert torch.equal(reps[0], sentinel)

    async def test_update_stores_decoded_prose(self):
        strategy = MagicMock()
        strategy.reason = AsyncMock(return_value=torch.zeros(4, 8))

        translator = MagicMock()
        translator.synthesize = AsyncMock(return_value="synthesized bucket prose")

        store = MagicMock()
        store.read.return_value = (_make_front_matter(), "old prose")
        store.write_prose = MagicMock()
        store.write_front_matter = MagicMock()

        lib = Librarian(
            "b1",
            store=store,
            registry=_make_mock_registry(),
            strategy=strategy,
            embedder=_make_mock_embedder(),
            translator=translator,
        )

        with patch("libucks.librarian._read_chunk_content", return_value="def foo(): pass"):
            await lib._handle_update(_make_update_event())

        store.write_prose.assert_called_once_with("b1", "synthesized bucket prose")

    async def test_update_without_translator_does_not_call_synthesize(self):
        strategy = MagicMock()
        strategy.reason = AsyncMock(return_value=torch.zeros(4, 8))

        store = MagicMock()
        store.read.return_value = (_make_front_matter(), "old prose")
        store.write_prose = MagicMock()
        store.write_front_matter = MagicMock()

        lib = Librarian(
            "b1",
            store=store,
            registry=_make_mock_registry(),
            strategy=strategy,
            embedder=_make_mock_embedder(),
            translator=None,
        )

        with patch("libucks.librarian._read_chunk_content", return_value="def foo(): pass"):
            await lib._handle_update(_make_update_event())

        # write_prose should still be called (with current prose unchanged)
        # but should NOT be called with a raw tensor string.
        written_prose = store.write_prose.call_args[0][1]
        assert "tensor(" not in written_prose

    async def test_update_prose_is_not_raw_tensor_string(self):
        """Regression: str(tensor) must never be written as prose."""
        strategy = MagicMock()
        strategy.reason = AsyncMock(return_value=torch.ones(4, 8))

        translator = MagicMock()
        translator.synthesize = AsyncMock(return_value="clean prose")

        store = MagicMock()
        store.read.return_value = (_make_front_matter(), "old prose")
        store.write_prose = MagicMock()
        store.write_front_matter = MagicMock()

        lib = Librarian(
            "b1",
            store=store,
            registry=_make_mock_registry(),
            strategy=strategy,
            embedder=_make_mock_embedder(),
            translator=translator,
        )

        with patch("libucks.librarian._read_chunk_content", return_value="def foo(): pass"):
            await lib._handle_update(_make_update_event())

        written_prose = store.write_prose.call_args[0][1]
        assert "tensor(" not in written_prose
        assert written_prose == "clean prose"


# ---------------------------------------------------------------------------
# _handle_tombstone
# ---------------------------------------------------------------------------

class TestLibrarianTombstoneProse:
    async def test_tombstone_calls_translator_synthesize_with_tensor(self):
        """When a chunk is removed but others survive, translator is called."""
        sentinel = torch.ones(4, 8)
        strategy = MagicMock()
        strategy.reason = AsyncMock(return_value=sentinel)

        translator = MagicMock()
        translator.synthesize = AsyncMock(return_value="tombstone prose")

        # Front matter has two chunks; we'll tombstone one, leaving one survivor.
        chunk_a = ChunkMetadata(
            chunk_id="c1", source_file="/f.py",
            start_line=1, end_line=3, git_sha="aa", token_count=50,
        )
        chunk_b = ChunkMetadata(
            chunk_id="c2", source_file="/f.py",
            start_line=4, end_line=6, git_sha="bb", token_count=50,
        )
        fm = BucketFrontMatter(
            bucket_id="b1", domain_label="auth",
            centroid_embedding="AAAA", token_count=100,
            chunks=[chunk_a, chunk_b],
        )

        store = MagicMock()
        store.read.return_value = (fm, "current prose")
        store.write_front_matter = MagicMock()
        store.write_prose = MagicMock()

        registry = _make_mock_registry()
        registry.get_all_centroids.return_value = {"b1": np.zeros(8, dtype=np.float32)}

        lib = Librarian(
            "b1",
            store=store,
            registry=registry,
            strategy=strategy,
            embedder=_make_mock_embedder(),
            translator=translator,
        )

        # Tombstone c1; c2 survives.
        event = _make_tombstone_event(chunk_ids=["c1"])
        await lib._handle_tombstone(event)

        translator.synthesize.assert_called_once()
        reps = translator.synthesize.call_args[0][1]
        assert len(reps) == 1
        assert torch.equal(reps[0], sentinel)

    async def test_tombstone_stores_decoded_prose(self):
        strategy = MagicMock()
        strategy.reason = AsyncMock(return_value=torch.zeros(4, 8))

        translator = MagicMock()
        translator.synthesize = AsyncMock(return_value="updated tombstone prose")

        chunk_a = ChunkMetadata(
            chunk_id="c1", source_file="/f.py",
            start_line=1, end_line=3, git_sha="aa", token_count=50,
        )
        chunk_b = ChunkMetadata(
            chunk_id="c2", source_file="/f.py",
            start_line=4, end_line=6, git_sha="bb", token_count=50,
        )
        fm = BucketFrontMatter(
            bucket_id="b1", domain_label="auth",
            centroid_embedding="AAAA", token_count=100,
            chunks=[chunk_a, chunk_b],
        )

        store = MagicMock()
        store.read.return_value = (fm, "current prose")
        store.write_front_matter = MagicMock()
        store.write_prose = MagicMock()

        registry = _make_mock_registry()
        registry.get_all_centroids.return_value = {"b1": np.zeros(8, dtype=np.float32)}

        lib = Librarian(
            "b1",
            store=store,
            registry=registry,
            strategy=strategy,
            embedder=_make_mock_embedder(),
            translator=translator,
        )

        event = _make_tombstone_event(chunk_ids=["c1"])
        await lib._handle_tombstone(event)

        store.write_prose.assert_called_once_with("b1", "updated tombstone prose")

    async def test_tombstone_without_translator_does_not_write_prose(self):
        """No translator → strategy is still called but write_prose is NOT called."""
        strategy = MagicMock()
        strategy.reason = AsyncMock(return_value=torch.zeros(4, 8))

        chunk_a = ChunkMetadata(
            chunk_id="c1", source_file="/f.py",
            start_line=1, end_line=3, git_sha="aa", token_count=50,
        )
        chunk_b = ChunkMetadata(
            chunk_id="c2", source_file="/f.py",
            start_line=4, end_line=6, git_sha="bb", token_count=50,
        )
        fm = BucketFrontMatter(
            bucket_id="b1", domain_label="auth",
            centroid_embedding="AAAA", token_count=100,
            chunks=[chunk_a, chunk_b],
        )

        store = MagicMock()
        store.read.return_value = (fm, "current prose")
        store.write_front_matter = MagicMock()
        store.write_prose = MagicMock()

        registry = _make_mock_registry()
        registry.get_all_centroids.return_value = {"b1": np.zeros(8, dtype=np.float32)}

        lib = Librarian(
            "b1",
            store=store,
            registry=registry,
            strategy=strategy,
            embedder=_make_mock_embedder(),
            translator=None,
        )

        event = _make_tombstone_event(chunk_ids=["c1"])
        await lib._handle_tombstone(event)

        store.write_prose.assert_not_called()
