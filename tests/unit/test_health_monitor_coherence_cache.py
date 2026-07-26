"""CM-B bug sweep: HealthMonitor must not re-embed the whole repo every 5 minutes.

`_compute_coherence` called `embedder.embed_batch(contents)` directly, with no
caching, once per bucket per health pass. HealthMonitor runs every 300s, so a
159-bucket repo re-embedded every chunk it owns twelve times an hour, forever.
Measured on this machine: the libucks-on-libucks server sat at 58% CPU while
the distill job it was competing with got 8%.

`ChunkRetriever` already solves exactly this — a (chunk_id, git_sha)-keyed
embedding cache, in memory and on disk, that re-embeds only chunks whose
content actually changed. HealthMonitor simply was never given one.

The subtlety that makes this more than a wiring change: the two components use
DIFFERENT `_read_chunk_content` variants (see test_read_chunk_content_families).
ChunkRetriever is CONTENT-family and drops chunks whose source file is
unreadable; HealthMonitor is GEOMETRY-family and embeds the file path instead,
deliberately, so dead chunks do not all collapse onto one degenerate vector.
Reusing the cache naively would silently change what a "dead" chunk contributes
to coherence, and therefore change split decisions.

So: cached embeddings for every chunk the cache knows, and the geometry path
fallback for every chunk it does not. Coherence is unchanged; the work is not.
"""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock

from libucks.health_monitor import HealthMonitor
from libucks.models.bucket import BucketFrontMatter
from libucks.models.chunk import ChunkMetadata


def _chunk(cid: str, source_file: str, sha: str = "sha1") -> ChunkMetadata:
    return ChunkMetadata(
        chunk_id=cid, source_file=source_file, start_line=1, end_line=2,
        git_sha=sha, token_count=10,
    )


def _fm(chunks: list[ChunkMetadata]) -> BucketFrontMatter:
    return BucketFrontMatter(
        bucket_id="b1", domain_label="d", centroid_embedding="AAAA",
        token_count=100, chunks=chunks,
    )


def _unit(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _monitor(store, embedder, retriever=None) -> HealthMonitor:
    return HealthMonitor(
        registry=MagicMock(), store=store, mitosis_service=MagicMock(),
        merging_service=MagicMock(), embedder=embedder,
        chunk_retriever=retriever,
    )


@pytest.fixture
def two_readable(tmp_path):
    files = []
    for i in range(2):
        f = tmp_path / f"m{i}.py"
        f.write_text(f"def f{i}():\n    return {i}\n")
        files.append(str(f))
    chunks = [_chunk(f"c{i}", files[i]) for i in range(2)]
    store = MagicMock()
    store.read.return_value = (_fm(chunks), "prose")
    return store, chunks


class TestUsesTheCacheWhenGivenOne:
    def test_cached_chunks_are_not_re_embedded(self, two_readable):
        store, chunks = two_readable
        embedder = MagicMock()
        retriever = MagicMock()
        retriever.embeddings_for.return_value = {
            "c0": _unit(0), "c1": _unit(1),
        }

        score = _monitor(store, embedder, retriever)._compute_coherence("b1")

        assert score is not None
        embedder.embed_batch.assert_not_called()
        embedder.embed.assert_not_called()
        retriever.embeddings_for.assert_called_once()

    def test_coherence_matches_the_uncached_computation(self, two_readable):
        """Same vectors in, same number out — caching must be invisible."""
        store, chunks = two_readable
        vecs = {"c0": _unit(0), "c1": _unit(1)}

        retriever = MagicMock()
        retriever.embeddings_for.return_value = vecs
        cached = _monitor(store, MagicMock(), retriever)._compute_coherence("b1")

        plain_embedder = MagicMock()
        plain_embedder.embed_batch.return_value = np.stack([vecs["c0"], vecs["c1"]])
        uncached = _monitor(store, plain_embedder)._compute_coherence("b1")

        assert cached == pytest.approx(uncached, abs=1e-6)


class TestGeometryFallbackSurvives:
    def test_chunk_missing_from_cache_is_embedded_as_its_path(self, tmp_path):
        """ChunkRetriever drops unreadable chunks; coherence must not.

        Without the fallback the dead chunk vanishes from the mean and a bucket
        full of deleted files would look perfectly coherent.
        """
        good = tmp_path / "ok.py"
        good.write_text("def ok():\n    pass\n")
        dead_path = "/gone/deleted.py"
        chunks = [_chunk("c0", str(good)), _chunk("c1", dead_path)]
        store = MagicMock()
        store.read.return_value = (_fm(chunks), "prose")

        embedder = MagicMock()
        embedder.embed.return_value = _unit(7)
        retriever = MagicMock()
        retriever.embeddings_for.return_value = {"c0": _unit(0)}  # c1 absent

        score = _monitor(store, embedder, retriever)._compute_coherence("b1")

        assert score is not None
        embedder.embed.assert_called_once()
        assert embedder.embed.call_args[0][0] == dead_path, (
            "the geometry fallback must embed the PATH, not empty string"
        )


class TestBackwardCompatibility:
    def test_without_a_retriever_it_still_uses_embed_batch(self, two_readable):
        """chunk_retriever is optional; existing callers must keep working."""
        store, _ = two_readable
        embedder = MagicMock()
        embedder.embed_batch.return_value = np.stack([_unit(0), _unit(1)])

        score = _monitor(store, embedder)._compute_coherence("b1")

        assert score is not None
        embedder.embed_batch.assert_called_once()

    def test_single_chunk_bucket_is_trivially_coherent(self, tmp_path):
        f = tmp_path / "one.py"
        f.write_text("x = 1\n")
        store = MagicMock()
        store.read.return_value = (_fm([_chunk("c0", str(f))]), "p")
        retriever = MagicMock()
        assert _monitor(store, MagicMock(), retriever)._compute_coherence("b1") == 1.0
        retriever.embeddings_for.assert_not_called()

    def test_missing_bucket_returns_none(self):
        store = MagicMock()
        store.read.side_effect = FileNotFoundError
        assert _monitor(store, MagicMock(), MagicMock())._compute_coherence("b1") is None

    def test_cache_failure_falls_back_instead_of_crashing(self, two_readable):
        """A broken cache must degrade to the old path, not kill the monitor."""
        store, _ = two_readable
        retriever = MagicMock()
        retriever.embeddings_for.side_effect = RuntimeError("cache exploded")
        embedder = MagicMock()
        embedder.embed_batch.return_value = np.stack([_unit(0), _unit(1)])

        score = _monitor(store, embedder, retriever)._compute_coherence("b1")

        assert score is not None
        embedder.embed_batch.assert_called_once()


class TestBridgeScopeIsSound:
    """`_load_heavy` builds its objects inside a nested `_sync_setup()`.

    Anything constructed in there reaches the outer body only via the return
    tuple. Wiring chunk_retriever into HealthMonitor tripped exactly this: the
    reference was added outside the nested function while the tuple was not
    updated, which is a NameError at `libucks serve` time and invisible to the
    suite, because _load_heavy loads real models and is never unit-tested.

    This is a static check, so it costs nothing and covers the whole function.
    """

    def test_no_name_escapes_sync_setup_unreturned(self):
        import ast
        from pathlib import Path

        src = (Path(__file__).resolve().parents[2]
               / "libucks" / "mcp_bridge.py").read_text()
        tree = ast.parse(src)

        sync = load = None
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if n.name == "_sync_setup":
                    sync = n
                elif n.name == "_load_heavy":
                    load = n
        assert sync is not None and load is not None, "bridge structure changed"

        def bound(node):
            out = set()
            for m in ast.walk(node):
                if isinstance(m, ast.Name) and isinstance(m.ctx, ast.Store):
                    out.add(m.id)
                elif isinstance(m, ast.alias):
                    out.add((m.asname or m.name).split(".")[0])
            return out

        inner, outer = bound(sync), set()
        sync_lines = set(range(sync.lineno, sync.end_lineno + 1))
        for m in ast.walk(load):
            if isinstance(m, ast.Name) and isinstance(m.ctx, ast.Store):
                if m.lineno not in sync_lines:
                    outer.add(m.id)
            elif isinstance(m, ast.alias) and m.lineno not in sync_lines:
                outer.add((m.asname or m.name).split(".")[0])

        leaked = sorted(
            m.id for m in ast.walk(load)
            if isinstance(m, ast.Name) and isinstance(m.ctx, ast.Load)
            and m.lineno not in sync_lines
            and m.id in inner and m.id not in outer
            and not hasattr(__builtins__, m.id)
        )
        assert leaked == [], (
            "these names are built inside _sync_setup but used outside it and "
            f"are not in its return tuple — NameError at serve time: {leaked}"
        )

    def test_health_monitor_receives_the_chunk_retriever(self):
        """The wiring itself, so a silent revert to uncached is caught."""
        import ast
        from pathlib import Path

        src = (Path(__file__).resolve().parents[2]
               / "libucks" / "mcp_bridge.py").read_text()
        calls = [
            n for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == "HealthMonitor"
        ]
        assert len(calls) == 1, f"expected one HealthMonitor(), found {len(calls)}"
        kwargs = {k.arg for k in calls[0].keywords}
        assert "chunk_retriever" in kwargs, (
            "HealthMonitor lost its cache — coherence will re-embed every "
            f"chunk of every bucket on every tick. Got: {sorted(kwargs)}"
        )


class TestChunkRetrieverExposesTheCache:
    def test_embeddings_for_returns_chunk_id_to_vector(self, tmp_path):
        from libucks.chunk_retriever import ChunkRetriever

        f = tmp_path / "a.py"
        f.write_text("def a():\n    return 1\n")
        chunks = [_chunk("c0", str(f))]
        store = MagicMock()
        store.read.return_value = (_fm(chunks), "p")
        embedder = MagicMock()
        embedder.embed.return_value = _unit(3)

        r = ChunkRetriever(cache_dir=tmp_path / "cache", embedder=embedder, store=store)
        out = r.embeddings_for("b1", chunks)

        assert set(out) == {"c0"}
        assert np.allclose(out["c0"], _unit(3))

    def test_second_call_hits_the_cache(self, tmp_path):
        from libucks.chunk_retriever import ChunkRetriever

        f = tmp_path / "a.py"
        f.write_text("def a():\n    return 1\n")
        chunks = [_chunk("c0", str(f))]
        store = MagicMock()
        store.read.return_value = (_fm(chunks), "p")
        embedder = MagicMock()
        embedder.embed.return_value = _unit(3)

        r = ChunkRetriever(cache_dir=tmp_path / "cache", embedder=embedder, store=store)
        r.embeddings_for("b1", chunks)
        r.embeddings_for("b1", chunks)

        assert embedder.embed.call_count == 1, "second pass must not re-embed"

    def test_changed_git_sha_forces_a_re_embed(self, tmp_path):
        from libucks.chunk_retriever import ChunkRetriever

        f = tmp_path / "a.py"
        f.write_text("def a():\n    return 1\n")
        store = MagicMock()
        embedder = MagicMock()
        embedder.embed.return_value = _unit(3)
        r = ChunkRetriever(cache_dir=tmp_path / "cache", embedder=embedder, store=store)

        r.embeddings_for("b1", [_chunk("c0", str(f), sha="old")])
        r.embeddings_for("b1", [_chunk("c0", str(f), sha="new")])

        assert embedder.embed.call_count == 2, "a content change must invalidate"
