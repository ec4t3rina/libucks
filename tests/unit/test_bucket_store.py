"""Phase 1 Testing Gate — test_bucket_store.py

Tests CRUD operations for BucketStore, with particular focus on the
front-matter / prose boundary: write_prose must not touch the YAML header,
and write_front_matter must not touch the prose body.
"""

import base64
import struct
from pathlib import Path

import numpy as np
import pytest
import yaml

from libucks.models.bucket import BucketFrontMatter
from libucks.models.chunk import ChunkMetadata
from libucks.storage.bucket_store import BucketStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path: Path) -> BucketStore:
    bucket_dir = tmp_path / ".libucks" / "buckets"
    bucket_dir.mkdir(parents=True)
    return BucketStore(bucket_dir)


def _centroid_b64() -> str:
    arr = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    return base64.b64encode(arr.tobytes()).decode()


def _make_chunks() -> list[ChunkMetadata]:
    return [
        ChunkMetadata(
            chunk_id="c001",
            source_file="src/auth/middleware.py",
            start_line=12,
            end_line=47,
            git_sha="e4f9a3b",
            token_count=312,
        )
    ]


PROSE = "## Auth Middleware\n\nHandles JWT validation on every request."


# ---------------------------------------------------------------------------
# create()
# ---------------------------------------------------------------------------

class TestCreate:
    def test_creates_md_file(self, store: BucketStore):
        path = store.create("a3f8c2d1", "auth middleware", _centroid_b64(), _make_chunks(), PROSE)
        assert path.exists()
        assert path.suffix == ".md"

    def test_file_path_contains_bucket_id(self, store: BucketStore):
        path = store.create("a3f8c2d1", "auth middleware", _centroid_b64(), _make_chunks(), PROSE)
        assert "a3f8c2d1" in path.name

    def test_front_matter_is_valid_yaml(self, store: BucketStore):
        store.create("a3f8c2d1", "auth middleware", _centroid_b64(), _make_chunks(), PROSE)
        raw = store._path("a3f8c2d1").read_text()
        # Strip the opening/closing --- delimiters and parse
        assert raw.startswith("---\n"), "File must start with YAML front-matter delimiter"
        parts = raw.split("---\n", 2)
        # parts[0] == '' (before first ---\n), parts[1] == yaml block, parts[2] == prose
        assert len(parts) == 3, "File must have exactly two --- delimiters"
        parsed = yaml.safe_load(parts[1])
        assert parsed["bucket_id"] == "a3f8c2d1"
        assert parsed["domain_label"] == "auth middleware"
        assert "centroid_embedding" in parsed
        assert isinstance(parsed["chunks"], list)
        assert parsed["chunks"][0]["chunk_id"] == "c001"

    def test_prose_body_is_present(self, store: BucketStore):
        store.create("a3f8c2d1", "auth middleware", _centroid_b64(), _make_chunks(), PROSE)
        _, prose = store.read("a3f8c2d1")
        assert prose.strip() == PROSE.strip()


# ---------------------------------------------------------------------------
# read()
# ---------------------------------------------------------------------------

class TestRead:
    def test_returns_correct_front_matter(self, store: BucketStore):
        centroid = _centroid_b64()
        store.create("a3f8c2d1", "auth middleware", centroid, _make_chunks(), PROSE)
        bfm, _ = store.read("a3f8c2d1")
        assert isinstance(bfm, BucketFrontMatter)
        assert bfm.bucket_id == "a3f8c2d1"
        assert bfm.domain_label == "auth middleware"
        assert bfm.centroid_embedding == centroid
        assert len(bfm.chunks) == 1
        assert bfm.chunks[0].chunk_id == "c001"

    def test_returns_correct_prose(self, store: BucketStore):
        store.create("a3f8c2d1", "auth middleware", _centroid_b64(), _make_chunks(), PROSE)
        _, prose = store.read("a3f8c2d1")
        assert prose.strip() == PROSE.strip()

    def test_nonexistent_bucket_raises(self, store: BucketStore):
        with pytest.raises(FileNotFoundError):
            store.read("does-not-exist")


# ---------------------------------------------------------------------------
# write_prose()
# ---------------------------------------------------------------------------

class TestWriteProse:
    def test_prose_is_updated(self, store: BucketStore):
        store.create("a3f8c2d1", "auth middleware", _centroid_b64(), _make_chunks(), PROSE)
        new_prose = "## Auth Middleware\n\nUpdated description of auth."
        store.write_prose("a3f8c2d1", new_prose)
        _, prose = store.read("a3f8c2d1")
        assert prose.strip() == new_prose.strip()

    def test_front_matter_preserved_byte_for_byte(self, store: BucketStore):
        store.create("a3f8c2d1", "auth middleware", _centroid_b64(), _make_chunks(), PROSE)
        raw_before = store._path("a3f8c2d1").read_text()
        # Extract YAML block before the write
        yaml_block_before = raw_before.split("---\n", 2)[1]

        store.write_prose("a3f8c2d1", "## New prose\n\nSomething entirely different.")

        raw_after = store._path("a3f8c2d1").read_text()
        yaml_block_after = raw_after.split("---\n", 2)[1]

        assert yaml_block_before == yaml_block_after, (
            "write_prose() must not alter the YAML front-matter block"
        )

    def test_nonexistent_bucket_raises(self, store: BucketStore):
        with pytest.raises(FileNotFoundError):
            store.write_prose("ghost", "some prose")


# ---------------------------------------------------------------------------
# write_front_matter()
# ---------------------------------------------------------------------------

class TestWriteFrontMatter:
    def test_front_matter_is_updated(self, store: BucketStore):
        store.create("a3f8c2d1", "auth middleware", _centroid_b64(), _make_chunks(), PROSE)
        bfm, _ = store.read("a3f8c2d1")
        updated = bfm.model_copy(update={"domain_label": "JWT validation", "token_count": 999})
        store.write_front_matter("a3f8c2d1", updated)
        bfm2, _ = store.read("a3f8c2d1")
        assert bfm2.domain_label == "JWT validation"
        assert bfm2.token_count == 999

    def test_prose_preserved_after_front_matter_update(self, store: BucketStore):
        store.create("a3f8c2d1", "auth middleware", _centroid_b64(), _make_chunks(), PROSE)
        bfm, _ = store.read("a3f8c2d1")
        updated = bfm.model_copy(update={"domain_label": "JWT validation"})
        store.write_front_matter("a3f8c2d1", updated)
        _, prose = store.read("a3f8c2d1")
        assert prose.strip() == PROSE.strip()

    def test_nonexistent_bucket_raises(self, store: BucketStore):
        bfm = BucketFrontMatter(
            bucket_id="ghost",
            domain_label="x",
            centroid_embedding=_centroid_b64(),
            token_count=0,
            chunks=[],
        )
        with pytest.raises(FileNotFoundError):
            store.write_front_matter("ghost", bfm)


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------

class TestDelete:
    def test_file_is_removed(self, store: BucketStore):
        store.create("a3f8c2d1", "auth middleware", _centroid_b64(), _make_chunks(), PROSE)
        assert store._path("a3f8c2d1").exists()
        store.delete("a3f8c2d1")
        assert not store._path("a3f8c2d1").exists()

    def test_read_after_delete_raises(self, store: BucketStore):
        store.create("a3f8c2d1", "auth middleware", _centroid_b64(), _make_chunks(), PROSE)
        store.delete("a3f8c2d1")
        with pytest.raises(FileNotFoundError):
            store.read("a3f8c2d1")

    def test_delete_nonexistent_raises(self, store: BucketStore):
        with pytest.raises(FileNotFoundError):
            store.delete("ghost")


# ---------------------------------------------------------------------------
# list_all()
# ---------------------------------------------------------------------------

class TestListAll:
    def test_empty_store_returns_empty_list(self, store: BucketStore):
        assert store.list_all() == []

    def test_returns_created_bucket_ids(self, store: BucketStore):
        store.create("bucket-a", "domain a", _centroid_b64(), _make_chunks(), "prose a")
        store.create("bucket-b", "domain b", _centroid_b64(), _make_chunks(), "prose b")
        ids = store.list_all()
        assert sorted(ids) == ["bucket-a", "bucket-b"]

    def test_deleted_bucket_not_returned(self, store: BucketStore):
        store.create("bucket-a", "domain a", _centroid_b64(), _make_chunks(), "prose a")
        store.create("bucket-b", "domain b", _centroid_b64(), _make_chunks(), "prose b")
        store.delete("bucket-a")
        assert store.list_all() == ["bucket-b"]

    def test_returns_exactly_created_count(self, store: BucketStore):
        for i in range(5):
            store.create(f"bucket-{i}", f"domain {i}", _centroid_b64(), [], f"prose {i}")
        assert len(store.list_all()) == 5
