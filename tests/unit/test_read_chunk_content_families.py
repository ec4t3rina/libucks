"""CM-B bug sweep: `_read_chunk_content` has two INTENTIONALLY different fallbacks.

Four modules define a function with this exact name and near-identical bodies.
Three return `""` when the source file is unreadable; `mitosis.py` returns the
file path instead. That looks like drift, and the obvious "cleanup" is to
collapse all four into one. Doing so would be a silent behaviour change to
bucket splitting, so the split is pinned here instead.

  GEOMETRY family — mitosis, and merging_service + health_monitor which import
  mitosis's copy. Output is fed to `embed_batch` and turned into centroids,
  k-means clusters, and coherence scores. Embedding "" yields a degenerate
  vector: every unreadable chunk lands on the same point, so deleted files
  would silently pull a bucket's centroid toward a meaningless location and
  distort every split/merge decision. Returning the path keeps some signal.

  CONTENT family — librarian, chunk_retriever (also used by translator), and
  data_generator. Output is shown to a model or a user as source text. A path
  masquerading as file content is a hallucination source, and every caller
  already skips falsy content, so "" is correct.

Both are right for their purpose. Neither is right for the other's.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from libucks.chunk_retriever import _read_chunk_content as content_variant
from libucks.health_monitor import _read_chunk_content as health_variant
from libucks.librarian import _read_chunk_content as librarian_variant
from libucks.merging_service import _read_chunk_content as merging_variant
from libucks.mitosis import _read_chunk_content as geometry_variant
from libucks.models.chunk import ChunkMetadata
from libucks.thinking.training.data_generator import _read_chunk_content as datagen_variant


def _chunk(source_file: str, start: int = 1, end: int = 3) -> ChunkMetadata:
    return ChunkMetadata(
        chunk_id="c1",
        source_file=source_file,
        start_line=start,
        end_line=end,
        git_sha="deadbeef",
        token_count=10,
    )


MISSING = "/nonexistent/definitely/not/here.py"

GEOMETRY = pytest.mark.parametrize(
    "fn", [geometry_variant, merging_variant, health_variant],
    ids=["mitosis", "merging_service", "health_monitor"],
)
CONTENT = pytest.mark.parametrize(
    "fn", [content_variant, librarian_variant, datagen_variant],
    ids=["chunk_retriever", "librarian", "data_generator"],
)


class TestGeometryFamilyFallsBackToPath:
    @GEOMETRY
    def test_missing_file_yields_the_path_not_empty(self, fn):
        """Empty string would collapse every dead chunk onto one embedding."""
        assert fn(_chunk(MISSING)) == MISSING

    @GEOMETRY
    def test_the_three_share_one_implementation(self, fn):
        assert fn is geometry_variant


class TestContentFamilyFallsBackToEmpty:
    @CONTENT
    def test_missing_file_yields_empty_not_the_path(self, fn):
        """A path returned as 'source code' is a hallucination source."""
        assert fn(_chunk(MISSING)) == ""


class TestBothFamiliesAgreeOnTheHappyPath:
    @pytest.fixture
    def sample(self, tmp_path: Path) -> str:
        f = tmp_path / "sample.py"
        f.write_text("line1\nline2\nline3\nline4\n")
        return str(f)

    @pytest.mark.parametrize(
        "fn",
        [geometry_variant, merging_variant, health_variant,
         content_variant, librarian_variant, datagen_variant],
    )
    def test_line_slicing_is_identical(self, fn, sample: str):
        """The divergence must be confined to the unreadable-file fallback."""
        assert fn(_chunk(sample, start=2, end=3)) == "line2\nline3"

    @pytest.mark.parametrize(
        "fn",
        [geometry_variant, merging_variant, health_variant,
         content_variant, librarian_variant, datagen_variant],
    )
    def test_start_line_is_one_indexed(self, fn, sample: str):
        assert fn(_chunk(sample, start=1, end=1)) == "line1"
