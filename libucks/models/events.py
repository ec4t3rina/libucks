from typing import List, Optional

from pydantic import BaseModel


class DiffHunk(BaseModel):
    file: str
    old_start: int
    old_end: int
    new_start: int
    new_end: int
    added_lines: List[str]
    removed_lines: List[str]


class DiffEvent(BaseModel):
    file: str
    hunks: List[DiffHunk]
    is_rename: bool
    old_path: Optional[str] = None
    new_path: Optional[str] = None


class UpdateEvent(BaseModel):
    bucket_id: str
    hunk: DiffHunk


class TombstoneEvent(BaseModel):
    chunk_ids: List[str]
    bucket_ids: List[str]


class PathUpdateEvent(BaseModel):
    old_path: str
    new_path: str


class QueryEvent(BaseModel):
    query: str
    bucket_id: str


class CreateBucketEvent(BaseModel):
    seed_content: str
