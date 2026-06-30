"""Phase 4-C — Cache augmentation package.

Implements DeepMind's "Deliberation in Latent Space via Differentiable Cache
Augmentation" (arXiv:2412.17747) adapted for libucks's multi-agent setting:

  - Frozen Qwen 2.5-3B receiver
  - Per-bucket KV cache precomputed at indexing time
  - Lightweight coprocessor reads bucket KV cache + learned soft tokens,
    emits z latent embeddings
  - Cross-bucket fusion combines z_b across k routed buckets
  - z_fused appended into receiver's KV cache at decode time

This file is the package marker; individual modules live alongside:
  - kv_extract.py      — capture past_key_values from frozen receiver
  - bucket_kv_cache.py — load/save/invalidate per-bucket caches
  - coprocessor.py     — coprocessor model (Phase 4-C.3)
  - fusion.py          — cross-bucket fusion block (Phase 4-C.4)
  - decode.py          — augmented-cache decode path (Phase 4-C.4)
"""

from libucks.cache_augmentation.bucket_kv_cache import BucketKVCache
from libucks.cache_augmentation.coprocessor import Coprocessor, CoprocessorBlock
from libucks.cache_augmentation.decode import augmented_decode, build_z_fused
from libucks.cache_augmentation.fusion import CrossBucketFusion
from libucks.cache_augmentation.kv_extract import extract_bucket_kv, restore_dynamic_cache

__all__ = [
    "BucketKVCache",
    "Coprocessor",
    "CoprocessorBlock",
    "CrossBucketFusion",
    "augmented_decode",
    "build_z_fused",
    "extract_bucket_kv",
    "restore_dynamic_cache",
]
