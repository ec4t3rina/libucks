"""Tests for AspectMapper — module affinity graph and import-aware clustering."""
from __future__ import annotations

import numpy as np
import pytest

from libucks.models.chunk import RawChunk


def _make_chunk(source_file: str, content: str = "", start: int = 1, end: int = 10) -> RawChunk:
    return RawChunk(
        source_file=source_file,
        start_line=start,
        end_line=end,
        content=content or f"# content from {source_file}",
        language="python",
    )


class TestAffinityMatrix:
    def test_same_file_chunks_score_higher_than_cross_file(self):
        """Two chunks from the same file should have higher affinity than cross-file."""
        chunks = [
            _make_chunk("/repo/auth.py"),
            _make_chunk("/repo/auth.py"),   # same file → bonus
            _make_chunk("/repo/utils.py"),  # different file → no bonus
        ]
        # Orthogonal embeddings so base cosine sim is 0 between all pairs
        embeddings = np.eye(3, 4, dtype=np.float32)

        from libucks.parsing.aspect_mapper import AspectMapper
        mapper = AspectMapper()
        affinity = mapper.compute_affinity_matrix(chunks, embeddings)

        assert affinity[0, 1] > affinity[0, 2]

    def test_import_linked_files_get_affinity_bonus(self):
        """If file A imports file B's stem, the A-B pair scores above unrelated pairs."""
        chunks = [
            _make_chunk("/repo/myapp.py", content="from mylib import helper\ndef foo(): pass"),
            _make_chunk("/repo/mylib.py", content="def helper(): pass"),   # imported by myapp
            _make_chunk("/repo/other.py", content="def unrelated(): pass"),  # not imported
        ]
        # Orthogonal: base cosine sim = 0 for all pairs
        embeddings = np.eye(3, 4, dtype=np.float32)

        from libucks.parsing.aspect_mapper import AspectMapper
        mapper = AspectMapper()
        affinity = mapper.compute_affinity_matrix(chunks, embeddings)

        assert affinity[0, 1] > affinity[0, 2]

    def test_affinity_matrix_is_symmetric(self):
        chunks = [_make_chunk(f"/repo/file{i}.py") for i in range(4)]
        embeddings = np.random.rand(4, 8).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings /= norms

        from libucks.parsing.aspect_mapper import AspectMapper
        affinity = AspectMapper().compute_affinity_matrix(chunks, embeddings)

        np.testing.assert_allclose(affinity, affinity.T, atol=1e-6)

    def test_affinity_values_clipped_to_0_1(self):
        """Same-file bonus + cosine sim 1.0 would exceed 1.0 without clipping."""
        chunks = [_make_chunk("/repo/a.py"), _make_chunk("/repo/a.py")]
        # Identical unit vectors → cosine sim = 1.0; adding same-file bonus clips to 1.0
        embeddings = np.ones((2, 4), dtype=np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        from libucks.parsing.aspect_mapper import AspectMapper
        affinity = AspectMapper().compute_affinity_matrix(chunks, embeddings)

        assert float(affinity.min()) >= 0.0
        assert float(affinity.max()) <= 1.0


class TestAspectMapperCluster:
    def test_same_file_chunks_cluster_together(self):
        """Aspect-aware clustering keeps same-file chunks in one cluster."""
        chunks = [
            _make_chunk("/repo/auth.py"),   # 0
            _make_chunk("/repo/auth.py"),   # 1
            _make_chunk("/repo/auth.py"),   # 2
            _make_chunk("/repo/db.py"),     # 3
            _make_chunk("/repo/db.py"),     # 4
        ]
        auth_embed = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        db_embed   = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        embeddings = np.stack([auth_embed] * 3 + [db_embed] * 2)

        from libucks.parsing.aspect_mapper import AspectMapper
        groups = AspectMapper().cluster(chunks, embeddings, n_clusters=2)

        all_indices = sorted(i for g in groups for i in g)
        assert all_indices == list(range(5))

        group_sets = [set(g) for g in groups]
        assert any({0, 1, 2}.issubset(s) for s in group_sets)
        assert any({3, 4}.issubset(s) for s in group_sets)

    def test_single_chunk_returns_one_group(self):
        chunks = [_make_chunk("/repo/a.py")]
        embeddings = np.ones((1, 4), dtype=np.float32)

        from libucks.parsing.aspect_mapper import AspectMapper
        groups = AspectMapper().cluster(chunks, embeddings, n_clusters=1)

        assert len(groups) == 1
        assert groups[0] == [0]

    def test_n_clusters_1_returns_all_in_one_group(self):
        chunks = [_make_chunk(f"/repo/file{i}.py") for i in range(5)]
        embeddings = np.random.rand(5, 4).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        from libucks.parsing.aspect_mapper import AspectMapper
        groups = AspectMapper().cluster(chunks, embeddings, n_clusters=1)

        assert len(groups) == 1
        assert sorted(groups[0]) == list(range(5))

    def test_empty_chunks_returns_empty(self):
        from libucks.parsing.aspect_mapper import AspectMapper
        groups = AspectMapper().cluster([], np.empty((0, 4)), n_clusters=1)
        assert groups == []
