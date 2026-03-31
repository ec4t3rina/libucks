"""Phase 1 Testing Gate — test_models.py

Tests round-trip serialization and Pydantic validation for all data contracts:
- ChunkMetadata
- BucketFrontMatter (including base64 centroid round-trip)
- All Event dataclasses
"""

import base64
import struct

import numpy as np
import pytest
from pydantic import ValidationError

from libucks.models.chunk import ChunkMetadata, RawChunk
from libucks.models.bucket import BucketFrontMatter
from libucks.models.events import (
    CreateBucketEvent,
    DiffEvent,
    DiffHunk,
    PathUpdateEvent,
    QueryEvent,
    TombstoneEvent,
    UpdateEvent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(**overrides) -> dict:
    base = dict(
        chunk_id="c001",
        source_file="src/auth/middleware.py",
        start_line=12,
        end_line=47,
        git_sha="e4f9a3b",
        token_count=312,
    )
    base.update(overrides)
    return base


def _make_bucket_front_matter(**overrides) -> dict:
    centroid = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    centroid_b64 = base64.b64encode(centroid.tobytes()).decode()
    base = dict(
        bucket_id="a3f8c2d1",
        domain_label="authentication middleware",
        centroid_embedding=centroid_b64,
        token_count=1842,
        chunks=[_make_chunk()],
    )
    base.update(overrides)
    return base


def _make_diff_hunk(**overrides) -> dict:
    base = dict(
        file="src/auth/middleware.py",
        old_start=10,
        old_end=15,
        new_start=10,
        new_end=20,
        added_lines=["def new_func():", "    pass"],
        removed_lines=["def old_func():", "    pass"],
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# ChunkMetadata
# ---------------------------------------------------------------------------

class TestChunkMetadata:
    def test_round_trip_dict(self):
        data = _make_chunk()
        chunk = ChunkMetadata(**data)
        result = chunk.model_dump()
        assert result == data

    def test_round_trip_json(self):
        data = _make_chunk()
        chunk = ChunkMetadata(**data)
        restored = ChunkMetadata.model_validate_json(chunk.model_dump_json())
        assert restored == chunk

    def test_missing_chunk_id_raises(self):
        data = _make_chunk()
        del data["chunk_id"]
        with pytest.raises(ValidationError):
            ChunkMetadata(**data)

    def test_missing_source_file_raises(self):
        data = _make_chunk()
        del data["source_file"]
        with pytest.raises(ValidationError):
            ChunkMetadata(**data)

    def test_missing_git_sha_raises(self):
        data = _make_chunk()
        del data["git_sha"]
        with pytest.raises(ValidationError):
            ChunkMetadata(**data)

    def test_token_count_must_be_non_negative(self):
        data = _make_chunk(token_count=-1)
        with pytest.raises(ValidationError):
            ChunkMetadata(**data)


# ---------------------------------------------------------------------------
# RawChunk
# ---------------------------------------------------------------------------

class TestRawChunk:
    def test_round_trip_dict(self):
        data = dict(
            source_file="src/auth/middleware.py",
            start_line=1,
            end_line=10,
            content="def foo(): pass",
            language="python",
        )
        chunk = RawChunk(**data)
        assert chunk.model_dump() == data

    def test_missing_content_raises(self):
        with pytest.raises(ValidationError):
            RawChunk(
                source_file="f.py",
                start_line=1,
                end_line=2,
                language="python",
                # content omitted
            )


# ---------------------------------------------------------------------------
# BucketFrontMatter
# ---------------------------------------------------------------------------

class TestBucketFrontMatter:
    def test_round_trip_dict(self):
        data = _make_bucket_front_matter()
        bfm = BucketFrontMatter(**data)
        result = bfm.model_dump()
        assert result["bucket_id"] == data["bucket_id"]
        assert result["domain_label"] == data["domain_label"]
        assert result["token_count"] == data["token_count"]
        assert len(result["chunks"]) == 1

    def test_centroid_base64_round_trip(self):
        original = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        encoded = base64.b64encode(original.tobytes()).decode()
        data = _make_bucket_front_matter(centroid_embedding=encoded)
        bfm = BucketFrontMatter(**data)

        # Decode back to numpy and compare
        decoded_bytes = base64.b64decode(bfm.centroid_embedding)
        n_floats = len(decoded_bytes) // 4
        restored = np.array(struct.unpack(f"{n_floats}f", decoded_bytes), dtype=np.float32)
        np.testing.assert_array_almost_equal(original, restored)

    def test_round_trip_json(self):
        data = _make_bucket_front_matter()
        bfm = BucketFrontMatter(**data)
        restored = BucketFrontMatter.model_validate_json(bfm.model_dump_json())
        assert restored == bfm

    def test_missing_bucket_id_raises(self):
        data = _make_bucket_front_matter()
        del data["bucket_id"]
        with pytest.raises(ValidationError):
            BucketFrontMatter(**data)

    def test_chunks_can_be_empty_list(self):
        data = _make_bucket_front_matter(chunks=[])
        bfm = BucketFrontMatter(**data)
        assert bfm.chunks == []

    def test_token_count_must_be_non_negative(self):
        data = _make_bucket_front_matter(token_count=-5)
        with pytest.raises(ValidationError):
            BucketFrontMatter(**data)


# ---------------------------------------------------------------------------
# DiffHunk
# ---------------------------------------------------------------------------

class TestDiffHunk:
    def test_round_trip(self):
        data = _make_diff_hunk()
        hunk = DiffHunk(**data)
        assert hunk.model_dump() == data

    def test_added_and_removed_can_be_empty(self):
        data = _make_diff_hunk(added_lines=[], removed_lines=[])
        hunk = DiffHunk(**data)
        assert hunk.added_lines == []
        assert hunk.removed_lines == []


# ---------------------------------------------------------------------------
# Event dataclasses
# ---------------------------------------------------------------------------

class TestDiffEvent:
    def test_round_trip_no_rename(self):
        hunk = DiffHunk(**_make_diff_hunk())
        event = DiffEvent(
            file="src/auth/middleware.py",
            hunks=[hunk],
            is_rename=False,
        )
        data = event.model_dump()
        restored = DiffEvent(**data)
        assert restored == event

    def test_round_trip_with_rename(self):
        hunk = DiffHunk(**_make_diff_hunk())
        event = DiffEvent(
            file="src/auth/new_middleware.py",
            hunks=[hunk],
            is_rename=True,
            old_path="src/auth/middleware.py",
            new_path="src/auth/new_middleware.py",
        )
        restored = DiffEvent.model_validate_json(event.model_dump_json())
        assert restored.is_rename is True
        assert restored.old_path == "src/auth/middleware.py"
        assert restored.new_path == "src/auth/new_middleware.py"

    def test_missing_file_raises(self):
        with pytest.raises(ValidationError):
            DiffEvent(hunks=[], is_rename=False)


class TestUpdateEvent:
    def test_round_trip(self):
        hunk = DiffHunk(**_make_diff_hunk())
        event = UpdateEvent(bucket_id="a3f8c2d1", hunk=hunk)
        restored = UpdateEvent.model_validate_json(event.model_dump_json())
        assert restored == event

    def test_missing_bucket_id_raises(self):
        hunk = DiffHunk(**_make_diff_hunk())
        with pytest.raises(ValidationError):
            UpdateEvent(hunk=hunk)


class TestTombstoneEvent:
    def test_round_trip(self):
        event = TombstoneEvent(
            chunk_ids=["c001", "c002"],
            bucket_ids=["a3f8c2d1"],
        )
        restored = TombstoneEvent.model_validate_json(event.model_dump_json())
        assert restored == event

    def test_missing_chunk_ids_raises(self):
        with pytest.raises(ValidationError):
            TombstoneEvent(bucket_ids=["x"])


class TestPathUpdateEvent:
    def test_round_trip(self):
        event = PathUpdateEvent(
            old_path="src/auth/middleware.py",
            new_path="src/auth/new_middleware.py",
        )
        restored = PathUpdateEvent.model_validate_json(event.model_dump_json())
        assert restored == event

    def test_missing_old_path_raises(self):
        with pytest.raises(ValidationError):
            PathUpdateEvent(new_path="src/auth/new.py")


class TestQueryEvent:
    def test_round_trip(self):
        event = QueryEvent(query="how does auth work?", bucket_id="a3f8c2d1")
        restored = QueryEvent.model_validate_json(event.model_dump_json())
        assert restored == event

    def test_missing_query_raises(self):
        with pytest.raises(ValidationError):
            QueryEvent(bucket_id="a3f8c2d1")


class TestCreateBucketEvent:
    def test_round_trip(self):
        event = CreateBucketEvent(seed_content="def authenticate(token): ...")
        restored = CreateBucketEvent.model_validate_json(event.model_dump_json())
        assert restored == event

    def test_missing_seed_content_raises(self):
        with pytest.raises(ValidationError):
            CreateBucketEvent()
