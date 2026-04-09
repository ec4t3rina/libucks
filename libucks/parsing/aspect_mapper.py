"""AspectMapper — module affinity graph for aspect-aware bucket clustering.

Computes a pairwise affinity matrix that blends cosine similarity with
two structural bonuses:
  +0.4  same-file  — chunks from the same source file belong together
  +0.2  import-link — file A imports file B's stem (or vice versa)

The resulting distance matrix (1 - affinity) is fed to agglomerative
clustering so that logically related chunks stay in the same bucket even
when their surface text embeddings diverge.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

from libucks.models.chunk import RawChunk

_SAME_FILE_BONUS: float = 0.4
_IMPORT_LINK_BONUS: float = 0.2


def _extract_import_stems(content: str) -> Set[str]:
    """Return the set of top-level module stems imported in a Python source string."""
    stems: Set[str] = set()
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return stems

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                stems.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                stems.add(node.module.split(".")[0])
    return stems


class AspectMapper:
    """Builds a module-affinity distance matrix for aspect-aware clustering."""

    def compute_affinity_matrix(
        self,
        chunks: List[RawChunk],
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """Return an N×N float32 affinity matrix with values in [0, 1].

        Base score is cosine similarity (dot product; embeddings assumed L2-normalized).
        Structural bonuses are added for same-file and import-linked pairs, then
        the result is clipped to [0, 1].
        """
        n = len(chunks)
        affinity = (embeddings @ embeddings.T).astype(np.float32)  # (N, N)

        files: List[str] = [c.source_file for c in chunks]

        # Compute import stems per unique file (Python only).
        import_stems: Dict[str, Set[str]] = {}
        for chunk in chunks:
            f = chunk.source_file
            if f not in import_stems:
                import_stems[f] = (
                    _extract_import_stems(chunk.content) if f.endswith(".py") else set()
                )

        file_stems: Dict[str, str] = {f: Path(f).stem for f in import_stems}

        for i in range(n):
            for j in range(i + 1, n):
                fi, fj = files[i], files[j]
                if fi == fj:
                    bonus = _SAME_FILE_BONUS
                else:
                    stem_i = file_stems.get(fi, "")
                    stem_j = file_stems.get(fj, "")
                    si = import_stems.get(fi, set())
                    sj = import_stems.get(fj, set())
                    bonus = _IMPORT_LINK_BONUS if (stem_j in si or stem_i in sj) else 0.0

                if bonus > 0.0:
                    affinity[i, j] += bonus
                    affinity[j, i] += bonus

        return np.clip(affinity, 0.0, 1.0)

    def cluster(
        self,
        chunks: List[RawChunk],
        embeddings: np.ndarray,
        n_clusters: int,
    ) -> List[List[int]]:
        """Cluster chunks into n_clusters groups using affinity-based distances.

        Returns a list of lists of chunk indices (one list per cluster).
        """
        n = len(chunks)
        if n == 0:
            return []
        if n_clusters <= 1 or n <= n_clusters:
            return [list(range(n))]

        affinity = self.compute_affinity_matrix(chunks, embeddings)
        distance = 1.0 - affinity
        np.fill_diagonal(distance, 0.0)

        from scipy.spatial.distance import squareform  # type: ignore[import-untyped]
        from scipy.cluster.hierarchy import fcluster, linkage  # type: ignore[import-untyped]

        condensed = squareform(distance, checks=False)
        Z = linkage(condensed, method="average")
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")

        groups: Dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            groups.setdefault(int(label), []).append(idx)
        return list(groups.values())
