"""Phase D: MergingService — translator injection for merged bucket prose.

Verifies that:
  - _merge() passes tensor from strategy.reason() through translator.synthesize()
    to produce the human-readable merged bucket prose.
  - When translator is None, merged prose falls back without writing str(tensor).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from unittest.mock import AsyncMock, MagicMock

from libucks.merging_service import MergingService
from libucks.models.bucket import BucketFrontMatter
from libucks.models.chunk import ChunkMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(chunk_id: str) -> ChunkMetadata:
    return ChunkMetadata(
        chunk_id=chunk_id,
        source_file="/fake/a.py",
        start_line=1,
        end_line=5,
        git_sha="abc",
        token_count=200,
    )


def _make_fm(bucket_id: str, domain: str, chunks: list[ChunkMetadata]) -> BucketFrontMatter:
    return BucketFrontMatter(
        bucket_id=bucket_id,
        domain_label=domain,
        centroid_embedding="AAAA",
        token_count=sum(c.token_count for c in chunks),
        chunks=chunks,
    )


def _make_registry(a: str = "a1", b: str = "b1") -> MagicMock:
    registry = MagicMock()
    # Two buckets above merge similarity, under token limit
    dim = 8
    v = np.zeros(dim, dtype=np.float32)
    v[0] = 1.0
    registry.get_all_centroids.return_value = {a: v, b: v.copy()}
    registry.get_token_count.return_value = 200  # well under 15k limit
    registry.register = AsyncMock()
    registry.deregister = AsyncMock()
    registry.save = MagicMock()
    registry._meta = {}
    return registry


def _make_store(fm_a: BucketFrontMatter, fm_b: BucketFrontMatter) -> MagicMock:
    store = MagicMock()
    store.read.side_effect = lambda bid: (fm_a, "prose_a") if bid == fm_a.bucket_id else (fm_b, "prose_b")
    store.create = MagicMock()
    store.delete = MagicMock()
    return store


def _make_agent() -> MagicMock:
    agent = MagicMock()
    agent.unregister_librarian = MagicMock()
    return agent


def _make_embedder(n_chunks: int = 2, dim: int = 8) -> MagicMock:
    embedder = MagicMock()
    embedder.embed_batch.return_value = np.ones((n_chunks, dim), dtype=np.float32)
    return embedder


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestMergingTranslatorConstruction:
    def test_translator_defaults_to_none(self):
        svc = MergingService(
            registry=_make_registry(),
            store=MagicMock(),
            agent=_make_agent(),
            embedder=_make_embedder(),
            strategy=MagicMock(),
        )
        assert svc._translator is None

    def test_translator_stored_when_injected(self):
        translator = MagicMock()
        svc = MergingService(
            registry=_make_registry(),
            store=MagicMock(),
            agent=_make_agent(),
            embedder=_make_embedder(),
            strategy=MagicMock(),
            translator=translator,
        )
        assert svc._translator is translator


# ---------------------------------------------------------------------------
# _merge() prose generation
# ---------------------------------------------------------------------------

class TestMergingProse:
    async def test_merge_calls_translator_synthesize_with_tensor(self):
        sentinel = torch.ones(4, 8)
        strategy = MagicMock()
        strategy.reason = AsyncMock(return_value=sentinel)

        translator = MagicMock()
        translator.synthesize = AsyncMock(return_value="merged prose")

        fm_a = _make_fm("a1", "auth", [_make_chunk("c1")])
        fm_b = _make_fm("b1", "core", [_make_chunk("c2")])

        svc = MergingService(
            registry=_make_registry("a1", "b1"),
            store=_make_store(fm_a, fm_b),
            agent=_make_agent(),
            embedder=_make_embedder(n_chunks=2),
            strategy=strategy,
            translator=translator,
        )

        await svc._merge("a1", "b1")

        translator.synthesize.assert_called_once()
        reps = translator.synthesize.call_args[0][1]
        assert len(reps) == 1
        assert torch.equal(reps[0], sentinel)

    async def test_merge_stores_decoded_prose(self):
        strategy = MagicMock()
        strategy.reason = AsyncMock(return_value=torch.zeros(4, 8))

        translator = MagicMock()
        translator.synthesize = AsyncMock(return_value="decoded merge prose")

        fm_a = _make_fm("a1", "auth", [_make_chunk("c1")])
        fm_b = _make_fm("b1", "core", [_make_chunk("c2")])
        store = _make_store(fm_a, fm_b)

        svc = MergingService(
            registry=_make_registry("a1", "b1"),
            store=store,
            agent=_make_agent(),
            embedder=_make_embedder(n_chunks=2),
            strategy=strategy,
            translator=translator,
        )

        await svc._merge("a1", "b1")

        create_call = store.create.call_args
        prose_arg = create_call.kwargs.get("prose") or create_call[1].get("prose")
        assert prose_arg == "decoded merge prose"

    async def test_merge_prose_is_not_raw_tensor_string(self):
        """Regression: str(tensor) must never be stored as bucket prose."""
        strategy = MagicMock()
        strategy.reason = AsyncMock(return_value=torch.ones(4, 8))

        translator = MagicMock()
        translator.synthesize = AsyncMock(return_value="clean merge prose")

        fm_a = _make_fm("a1", "auth", [_make_chunk("c1")])
        fm_b = _make_fm("b1", "core", [_make_chunk("c2")])
        store = _make_store(fm_a, fm_b)

        svc = MergingService(
            registry=_make_registry("a1", "b1"),
            store=store,
            agent=_make_agent(),
            embedder=_make_embedder(n_chunks=2),
            strategy=strategy,
            translator=translator,
        )

        await svc._merge("a1", "b1")

        create_call = store.create.call_args
        prose_arg = create_call.kwargs.get("prose") or create_call[1].get("prose")
        assert "tensor(" not in prose_arg

    async def test_merge_without_translator_does_not_write_tensor_as_prose(self):
        strategy = MagicMock()
        strategy.reason = AsyncMock(return_value=torch.ones(4, 8))

        fm_a = _make_fm("a1", "auth", [_make_chunk("c1")])
        fm_b = _make_fm("b1", "core", [_make_chunk("c2")])
        store = _make_store(fm_a, fm_b)

        svc = MergingService(
            registry=_make_registry("a1", "b1"),
            store=store,
            agent=_make_agent(),
            embedder=_make_embedder(n_chunks=2),
            strategy=strategy,
            translator=None,
        )

        await svc._merge("a1", "b1")

        create_call = store.create.call_args
        prose_arg = create_call.kwargs.get("prose") or create_call[1].get("prose")
        assert "tensor(" not in prose_arg
