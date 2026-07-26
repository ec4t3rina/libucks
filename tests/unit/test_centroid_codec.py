"""CM-B bug sweep: exactly one centroid encoder, and it round-trips.

`_encode_centroid` existed as six byte-identical copies — in bucket_registry,
librarian, mitosis, merging_service, init_orchestrator and novel_bucket_service
— while the matching `_decode_centroid` existed in exactly one place
(bucket_registry). Five modules wrote a wire format that a single other module
was responsible for reading, with nothing tying them together.

Nothing had drifted yet. But this is the same shape as the two duplication bugs
already found in this sweep (`_read_chunk_content`, and the four teacher-Q&A
call sites where one was fixed and three were not): a change to the reader
would silently corrupt every centroid written by the five writers, and every
bucket's routing with it.

These tests pin the invariant rather than the current text, so a re-introduced
copy fails here instead of in production months later.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from libucks.storage.bucket_registry import decode_centroid, encode_centroid

_LIBUCKS = Path(__file__).resolve().parents[2] / "libucks"


class TestSingleDefinition:
    def test_only_bucket_registry_defines_an_encoder(self):
        """No module may grow its own copy of the centroid encoder."""
        offenders = []
        for path in sorted(_LIBUCKS.rglob("*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef):
                    continue
                if node.name not in ("encode_centroid", "_encode_centroid",
                                     "decode_centroid", "_decode_centroid"):
                    continue
                if path.name != "bucket_registry.py":
                    offenders.append(f"{path.relative_to(_LIBUCKS)}:{node.lineno} {node.name}")
        assert offenders == [], (
            "centroid codec re-implemented outside bucket_registry.py — import it "
            f"instead: {offenders}"
        )

    def test_the_known_former_duplicators_still_encode(self):
        """The five modules must still reach an encoder, just not their own."""
        for mod in ("librarian", "mitosis", "merging_service",
                    "init_orchestrator", "novel_bucket_service"):
            src = (_LIBUCKS / f"{mod}.py").read_text()
            assert "encode_centroid" in src, f"{mod} lost its centroid encoding"


class TestRoundTrip:
    @pytest.mark.parametrize("dim", [8, 384, 768, 1024])
    def test_round_trip_preserves_values(self, dim: int):
        rng = np.random.default_rng(0)
        v = rng.standard_normal(dim).astype(np.float32)
        assert np.allclose(decode_centroid(encode_centroid(v)), v, atol=0.0)

    def test_round_trip_preserves_dimension(self):
        v = np.zeros(384, dtype=np.float32)
        assert decode_centroid(encode_centroid(v)).shape == (384,)

    def test_float64_input_is_narrowed_not_corrupted(self):
        """Callers pass float64 means; the codec must narrow, not reinterpret."""
        v = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        out = decode_centroid(encode_centroid(v))
        assert out.dtype == np.float32
        assert np.allclose(out, v, atol=1e-7)

    def test_normalisation_is_preserved(self):
        """Routing is cosine over these vectors — unit norm must survive."""
        rng = np.random.default_rng(1)
        v = rng.standard_normal(384).astype(np.float32)
        v /= np.linalg.norm(v)
        assert np.linalg.norm(decode_centroid(encode_centroid(v))) == pytest.approx(1.0, abs=1e-6)
