from typing import List, Optional

from pydantic import BaseModel, field_validator

from libucks.models.chunk import ChunkMetadata


class BucketFrontMatter(BaseModel):
    bucket_id: str
    domain_label: str
    centroid_embedding: str  # base64-encoded float32 array
    token_count: int
    chunks: List[ChunkMetadata]
    # Phase 6-A metadata — all optional so old bucket files deserialise without error
    last_indexed_at: Optional[str] = None   # ISO-8601 UTC of last Librarian write
    index_head_sha: Optional[str] = None    # git HEAD SHA at last Librarian write
    coherence_score: Optional[float] = None # set by HealthMonitor (Phase 6-E)
    parent_bucket_id: Optional[str] = None  # set on mitosis children
    generation: int = 0                     # incremented on each mitosis

    @field_validator("token_count")
    @classmethod
    def token_count_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("token_count must be >= 0")
        return v
