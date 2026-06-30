"""Per-bucket KV cache persistence on disk.

Layout:
    <repo>/.libucks/kv_cache/<bucket_id>.safetensors

Storage format: safetensors, contains the flat dict from kv_extract:
    layer_<i>_K, layer_<i>_V for i in [0, n_layers)
    _meta_seq_len, _meta_n_layers
Plus a small JSON metadata file for invalidation:
    <repo>/.libucks/kv_cache/<bucket_id>.json
    { "chunk_ids": [...], "git_sha_set": "<hash>", "model_id": "...",
      "max_tokens": <int>, "indexed_at": "<iso>" }

Invalidation is on chunk-set + git_sha changes: when a bucket's chunk
list or any contained chunk's git_sha changes (mitosis, merge, normal
update), we drop the cache and rebuild on next access.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file, save_file


def _log(msg: str) -> None:
    print(f"[libucks:cache_aug] {msg}", file=sys.stderr, flush=True)


def _chunk_set_signature(chunks: list) -> str:
    """Stable hash of (chunk_id, git_sha) pairs sorted by chunk_id. Detects
    chunk additions, removals, and content changes."""
    pairs = sorted([(c.chunk_id, c.git_sha) for c in chunks])
    h = hashlib.sha256()
    for cid, sha in pairs:
        h.update(cid.encode())
        h.update(b"\x00")
        h.update(sha.encode())
        h.update(b"\x01")
    return h.hexdigest()[:16]


class BucketKVCache:
    """Disk-backed per-bucket KV cache with content-based invalidation."""

    def __init__(self, cache_dir: Path, model_id: str, max_tokens: int = 1024) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._model_id = model_id
        self._max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def _tensor_path(self, bucket_id: str) -> Path:
        return self._cache_dir / f"{bucket_id}.safetensors"

    def _meta_path(self, bucket_id: str) -> Path:
        return self._cache_dir / f"{bucket_id}.json"

    # ------------------------------------------------------------------
    # Save / load / invalidate
    # ------------------------------------------------------------------

    def save(
        self,
        bucket_id: str,
        flat_kv: dict[str, torch.Tensor],
        chunks: list,
    ) -> None:
        """Persist a bucket's KV cache + metadata. `chunks` is the bucket's
        ChunkMetadata list (used to derive the invalidation signature)."""
        save_file(flat_kv, str(self._tensor_path(bucket_id)))
        meta = {
            "chunk_ids": sorted(c.chunk_id for c in chunks),
            "git_sha_set": _chunk_set_signature(chunks),
            "model_id": self._model_id,
            "max_tokens": self._max_tokens,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
        self._meta_path(bucket_id).write_text(json.dumps(meta, indent=2))
        _log(f"save: bucket={bucket_id[:10]} chunks={len(chunks)} sig={meta['git_sha_set']}")

    def load(
        self,
        bucket_id: str,
        chunks: list,
        device: Optional[torch.device] = None,
    ) -> Optional[dict[str, torch.Tensor]]:
        """Load a bucket's KV cache if it exists AND is fresh.

        Returns the flat tensor dict on hit, or None if the cache is missing
        or stale. Stale = chunk_set_signature mismatch OR model_id mismatch.
        """
        tp = self._tensor_path(bucket_id)
        mp = self._meta_path(bucket_id)
        if not tp.exists() or not mp.exists():
            return None
        try:
            meta = json.loads(mp.read_text())
        except Exception:
            return None
        if meta.get("model_id") != self._model_id:
            _log(f"load: bucket={bucket_id[:10]} stale (model_id changed)")
            return None
        if meta.get("git_sha_set") != _chunk_set_signature(chunks):
            _log(f"load: bucket={bucket_id[:10]} stale (chunk signature changed)")
            return None
        try:
            flat = load_file(str(tp))
        except Exception as exc:
            _log(f"load: bucket={bucket_id[:10]} read failed: {exc}")
            return None
        if device is not None:
            flat = {k: v.to(device) for k, v in flat.items()}
        return flat

    def invalidate(self, bucket_id: str) -> None:
        """Drop the cache files for a bucket (call on mitosis / merge / removal)."""
        for path in (self._tensor_path(bucket_id), self._meta_path(bucket_id)):
            if path.exists():
                path.unlink()
        _log(f"invalidate: bucket={bucket_id[:10]}")

    def exists(self, bucket_id: str) -> bool:
        return self._tensor_path(bucket_id).exists() and self._meta_path(bucket_id).exists()

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    @property
    def max_tokens(self) -> int:
        return self._max_tokens
