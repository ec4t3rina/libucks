from typing import Optional

from pydantic import BaseModel, field_validator


class ChunkMetadata(BaseModel):
    chunk_id: str
    source_file: str
    start_line: int
    end_line: int
    git_sha: str
    token_count: int
    indexed_at: Optional[str] = None  # ISO-8601 UTC; set by Librarian on every update

    @field_validator("token_count")
    @classmethod
    def token_count_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("token_count must be >= 0")
        return v


class RawChunk(BaseModel):
    source_file: str
    start_line: int
    end_line: int
    content: str
    language: str
