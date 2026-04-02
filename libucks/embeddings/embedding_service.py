"""EmbeddingService — singleton wrapper around sentence-transformers.

Design invariants:
  1. The SentenceTransformer model is loaded exactly once per process via
     get_instance().  Subsequent calls return the cached instance.
  2. Every vector returned by embed() and every row returned by embed_batch()
     is strictly L2-normalised (norm within 1e-6 of 1.0).  This is a hard
     contract: CentralAgent's cosine similarity is computed as dot(q, c_b),
     which equals cosine similarity only when both vectors are unit vectors.
  3. reset() exists exclusively for test isolation.  It must not be called
     in production code.
"""

from __future__ import annotations

from typing import ClassVar, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """L2-normalise a 1-D or 2-D float32 array in-place and return it."""
    mat = mat.astype(np.float32)
    if mat.ndim == 1:
        norm = np.linalg.norm(mat)
        if norm > 0:
            mat /= norm
    else:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)  # avoid division by zero
        mat /= norms
    return mat


class EmbeddingService:
    _instance: ClassVar[Optional["EmbeddingService"]] = None

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model = SentenceTransformer(model_name)

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls, model_name: str = _DEFAULT_MODEL) -> "EmbeddingService":
        if cls._instance is None:
            cls._instance = cls(model_name)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Discard the cached instance.  Test use only."""
        cls._instance = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str) -> np.ndarray:
        """Embed a single string and return a normalised float32 vector."""
        raw = self._model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return _l2_normalize(raw)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a list of strings and return a (N, D) normalised float32 matrix."""
        raw = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return _l2_normalize(raw)
