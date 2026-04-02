"""BucketRegistry — in-memory index of all active buckets.

Per-bucket state:
  centroid       np.ndarray (float32, L2-normalised)
  token_count    int
  lock           asyncio.Lock  (per-bucket write guard for Librarians)
  is_splitting   bool          (mitosis guard for CentralAgent)

Centroid is persisted to registry.json as a base64-encoded float32 array.
asyncio.Lock objects are never serialised — they are reconstructed on load().

All mutating public methods are async so callers can use asyncio.gather.
Since asyncio is single-threaded and none of the critical sections contain
an await point, all 100-concurrent-register calls are safe without an
additional registry-level lock.
"""

import asyncio
import base64
import json
import struct
from pathlib import Path
from typing import Dict, Optional

import numpy as np


class _BucketEntry:
    __slots__ = (
        "centroid", "token_count", "lock", "is_splitting",
        "last_indexed_at", "index_head_sha", "coherence_score",
        "query_hit_count", "last_query_hit_at",
    )

    def __init__(self, centroid: np.ndarray, token_count: int, lock: asyncio.Lock) -> None:
        self.centroid = centroid
        self.token_count = token_count
        self.lock = lock
        self.is_splitting = False
        self.last_indexed_at: Optional[str] = None
        self.index_head_sha: Optional[str] = None
        self.coherence_score: Optional[float] = None
        self.query_hit_count: int = 0
        self.last_query_hit_at: Optional[str] = None


def _encode_centroid(centroid: np.ndarray) -> str:
    return base64.b64encode(centroid.astype(np.float32).tobytes()).decode()


def _decode_centroid(encoded: str) -> np.ndarray:
    raw = base64.b64decode(encoded)
    n = len(raw) // 4
    return np.array(struct.unpack(f"{n}f", raw), dtype=np.float32)


class BucketRegistry:
    def __init__(self, registry_path: Path) -> None:
        self._path = registry_path
        self._buckets: Dict[str, _BucketEntry] = {}
        self._meta: Dict[str, object] = {
            "schema_version": 2,
            "last_indexed_head": None,
            "last_indexed_at": None,
            "watcher_pid": None,
            "merge_history": [],
        }

    # ------------------------------------------------------------------
    # Mutating operations (async so callers can use asyncio.gather)
    # ------------------------------------------------------------------

    async def register(
        self, bucket_id: str, centroid: np.ndarray, token_count: int
    ) -> None:
        if bucket_id in self._buckets:
            # Preserve the existing lock — replacing it would deadlock live holders.
            entry = self._buckets[bucket_id]
            entry.centroid = centroid.astype(np.float32)
            entry.token_count = token_count
        else:
            self._buckets[bucket_id] = _BucketEntry(
                centroid=centroid.astype(np.float32),
                token_count=token_count,
                lock=asyncio.Lock(),
            )

    async def deregister(self, bucket_id: str) -> None:
        if bucket_id not in self._buckets:
            raise KeyError(f"Bucket not registered: {bucket_id!r}")
        del self._buckets[bucket_id]

    async def set_splitting(self, bucket_id: str, flag: bool) -> None:
        if bucket_id not in self._buckets:
            raise KeyError(f"Bucket not registered: {bucket_id!r}")
        self._buckets[bucket_id].is_splitting = flag

    def update_index_timestamp(
        self, bucket_id: str, last_indexed_at: str, index_head_sha: str
    ) -> None:
        """Record when and at which git HEAD a bucket was last written by a Librarian."""
        if bucket_id not in self._buckets:
            raise KeyError(f"Bucket not registered: {bucket_id!r}")
        entry = self._buckets[bucket_id]
        entry.last_indexed_at = last_indexed_at
        entry.index_head_sha = index_head_sha

    # ------------------------------------------------------------------
    # Read-only operations (sync — safe because asyncio is single-threaded)
    # ------------------------------------------------------------------

    def get_all_centroids(self) -> Dict[str, np.ndarray]:
        return {bid: entry.centroid for bid, entry in self._buckets.items()}

    def get_token_count(self, bucket_id: str) -> int:
        if bucket_id not in self._buckets:
            raise KeyError(f"Bucket not registered: {bucket_id!r}")
        return self._buckets[bucket_id].token_count

    def is_splitting(self, bucket_id: str) -> bool:
        if bucket_id not in self._buckets:
            raise KeyError(f"Bucket not registered: {bucket_id!r}")
        return self._buckets[bucket_id].is_splitting

    def get_lock(self, bucket_id: str) -> asyncio.Lock:
        if bucket_id not in self._buckets:
            raise KeyError(f"Bucket not registered: {bucket_id!r}")
        return self._buckets[bucket_id].lock

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        data: Dict[str, object] = {"_meta": self._meta}
        for bucket_id, entry in self._buckets.items():
            data[bucket_id] = {
                "centroid_embedding": _encode_centroid(entry.centroid),
                "token_count": entry.token_count,
                "is_splitting": entry.is_splitting,
                "last_indexed_at": entry.last_indexed_at,
                "index_head_sha": entry.index_head_sha,
                "coherence_score": entry.coherence_score,
                "query_hit_count": entry.query_hit_count,
                "last_query_hit_at": entry.last_query_hit_at,
            }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self) -> None:
        if not self._path.exists():
            return
        data = json.loads(self._path.read_text(encoding="utf-8"))
        for key, state in data.items():
            if key.startswith("_"):
                # Handle registry-level meta blocks; ignore unknown underscore keys.
                if key == "_meta" and isinstance(state, dict):
                    self._meta.update(state)
                continue
            bucket_id = key
            centroid = _decode_centroid(state["centroid_embedding"])
            # Preserve any existing lock if already in memory (e.g. partial reload).
            existing = self._buckets.get(bucket_id)
            lock = existing.lock if existing else asyncio.Lock()
            entry = _BucketEntry(centroid=centroid, token_count=state["token_count"], lock=lock)
            entry.is_splitting = state.get("is_splitting", False)
            entry.last_indexed_at = state.get("last_indexed_at")
            entry.index_head_sha = state.get("index_head_sha")
            entry.coherence_score = state.get("coherence_score")
            entry.query_hit_count = state.get("query_hit_count", 0)
            entry.last_query_hit_at = state.get("last_query_hit_at")
            self._buckets[bucket_id] = entry
