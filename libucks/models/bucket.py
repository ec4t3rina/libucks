from typing import List

from pydantic import BaseModel, field_validator

from libucks.models.chunk import ChunkMetadata


class BucketFrontMatter(BaseModel):
    bucket_id: str
    domain_label: str
    centroid_embedding: str  # base64-encoded float32 array
    token_count: int
    chunks: List[ChunkMetadata]

    @field_validator("token_count")
    @classmethod
    def token_count_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("token_count must be >= 0")
        return v
