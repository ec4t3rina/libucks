"""Phase D: MitosisService — translator injection for child bucket prose.

Verifies that:
  - split() passes tensor from strategy.reason() through translator.synthesize()
    to produce human-readable child bucket prose.
  - Both child buckets receive decoded prose strings.
  - When translator is None, child prose falls back (no str(tensor) written).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from unittest.mock import AsyncMock, MagicMock, call

from libucks.mitosis import MitosisService
from libucks.models.bucket import BucketFrontMatter
from libucks.models.chunk import ChunkMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(chunk_id: str, source_file: str = "/fake/a.py") -> ChunkMetadata:
    return ChunkMetadata(
        chunk_id=chunk_id,
        source_file=source_file,
        start_line=1,
        end_line=5,
        git_sha="abc",
        token_count=100,
    )


def _make_front_matter(chunks: list[ChunkMetadata]) -> BucketFrontMatter:
    return BucketFrontMatter(
        bucket_id="parent",
        domain_label="utils",
        centroid_embedding="AAAA",
        token_count=sum(c.token_count for c in chunks),
        chunks=chunks,
    )


def _make_registry() -> MagicMock:
    registry = MagicMock()
    registry.set_splitting = AsyncMock()
    registry.register = AsyncMock()
    registry.deregister = AsyncMock()
    registry.save = MagicMock()
    return registry


def _make_agent() -> MagicMock:
    agent = MagicMock()
    agent.register_librarian = MagicMock()
    agent.unregister_librarian = MagicMock()
    agent.clear_splitting = AsyncMock()
    return agent


def _make_embedder(n_chunks: int = 4, dim: int = 8) -> MagicMock:
    embedder = MagicMock()
    embedder.embed_batch.return_value = np.random.rand(n_chunks, dim).astype(np.float32)
    return embedder


def _make_store(fm: BucketFrontMatter, prose: str = "existing prose") -> MagicMock:
    store = MagicMock()
    store.read.return_value = (fm, prose)
    store.create = MagicMock()
    store.delete = MagicMock()
    return store


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestMitosisTranslatorConstruction:
    def test_translator_defaults_to_none(self):
        svc = MitosisService(
            store=MagicMock(),
            registry=_make_registry(),
            embedder=_make_embedder(),
            agent=_make_agent(),
            strategy=MagicMock(),
        )
        assert svc._translator is None

    def test_translator_stored_when_injected(self):
        translator = MagicMock()
        svc = MitosisService(
            store=MagicMock(),
            registry=_make_registry(),
            embedder=_make_embedder(),
            agent=_make_agent(),
            strategy=MagicMock(),
            translator=translator,
        )
        assert svc._translator is translator


# ---------------------------------------------------------------------------
# split() prose generation
# ---------------------------------------------------------------------------

class TestMitosisChildProse:
    async def test_split_calls_translator_synthesize_for_each_child(self):
        """translator.synthesize() called exactly twice — once per child bucket."""
        sentinel = torch.ones(4, 8)
        strategy = MagicMock()
        strategy.reason = AsyncMock(return_value=sentinel)

        translator = MagicMock()
        translator.synthesize = AsyncMock(return_value="child prose")

        chunks = [_make_chunk(f"c{i}") for i in range(4)]
        fm = _make_front_matter(chunks)
        store = _make_store(fm)
        embedder = _make_embedder(n_chunks=4)
        # Force deterministic 2-cluster split by controlling embedder output
        # (first 2 chunks → cluster 0, last 2 → cluster 1)
        embedder.embed_batch.return_value = np.array(
            [[1.0, 0, 0, 0, 0, 0, 0, 0],
             [1.0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 1.0],
             [0, 0, 0, 0, 0, 0, 0, 1.0]],
            dtype=np.float32,
        )

        svc = MitosisService(
            store=store,
            registry=_make_registry(),
            embedder=embedder,
            agent=_make_agent(),
            strategy=strategy,
            translator=translator,
        )

        await svc.split("parent")

        assert translator.synthesize.call_count == 2

    async def test_split_child_prose_is_decoded_string(self):
        """Prose stored in child bucket must be the string from translator, not str(tensor)."""
        strategy = MagicMock()
        strategy.reason = AsyncMock(return_value=torch.zeros(4, 8))

        translator = MagicMock()
        translator.synthesize = AsyncMock(return_value="decoded child prose")

        chunks = [_make_chunk(f"c{i}") for i in range(4)]
        fm = _make_front_matter(chunks)
        store = _make_store(fm)
        store.create = MagicMock()
        embedder = _make_embedder(n_chunks=4)
        embedder.embed_batch.return_value = np.array(
            [[1.0, 0, 0, 0, 0, 0, 0, 0],
             [1.0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 1.0],
             [0, 0, 0, 0, 0, 0, 0, 1.0]],
            dtype=np.float32,
        )

        svc = MitosisService(
            store=store,
            registry=_make_registry(),
            embedder=embedder,
            agent=_make_agent(),
            strategy=strategy,
            translator=translator,
        )

        await svc.split("parent")

        # Both store.create() calls should have prose="decoded child prose"
        for c in store.create.call_args_list:
            assert c.kwargs["prose"] == "decoded child prose" or c[1]["prose"] == "decoded child prose"

    async def test_split_without_translator_does_not_write_tensor_as_prose(self):
        """No translator → child prose must never be str(tensor)."""
        strategy = MagicMock()
        strategy.reason = AsyncMock(return_value=torch.ones(4, 8))

        chunks = [_make_chunk(f"c{i}") for i in range(4)]
        fm = _make_front_matter(chunks)
        store = _make_store(fm)
        store.create = MagicMock()
        embedder = _make_embedder(n_chunks=4)
        embedder.embed_batch.return_value = np.array(
            [[1.0, 0, 0, 0, 0, 0, 0, 0],
             [1.0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 1.0],
             [0, 0, 0, 0, 0, 0, 0, 1.0]],
            dtype=np.float32,
        )

        svc = MitosisService(
            store=store,
            registry=_make_registry(),
            embedder=embedder,
            agent=_make_agent(),
            strategy=strategy,
            translator=None,
        )

        await svc.split("parent")

        for c in store.create.call_args_list:
            prose_arg = c.kwargs.get("prose") or c[1].get("prose", "")
            assert "tensor(" not in prose_arg

    async def test_split_passes_translator_to_child_librarians(self):
        """Newly created child Librarians must receive the translator instance."""
        strategy = MagicMock()
        strategy.reason = AsyncMock(return_value=torch.zeros(4, 8))

        translator = MagicMock()
        translator.synthesize = AsyncMock(return_value="child prose")

        chunks = [_make_chunk(f"c{i}") for i in range(4)]
        fm = _make_front_matter(chunks)
        store = _make_store(fm)
        embedder = _make_embedder(n_chunks=4)
        embedder.embed_batch.return_value = np.array(
            [[1.0, 0, 0, 0, 0, 0, 0, 0],
             [1.0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 1.0],
             [0, 0, 0, 0, 0, 0, 0, 1.0]],
            dtype=np.float32,
        )

        agent = _make_agent()

        svc = MitosisService(
            store=store,
            registry=_make_registry(),
            embedder=embedder,
            agent=agent,
            strategy=strategy,
            translator=translator,
        )

        await svc.split("parent")

        assert agent.register_librarian.call_count == 2
        for c in agent.register_librarian.call_args_list:
            librarian = c[0][1]  # second positional arg is the Librarian instance
            assert librarian._translator is translator
